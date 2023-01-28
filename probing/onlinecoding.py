import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--FIXED", action="store_true")
parser.add_argument("--LAYER", type=int)
parser.add_argument("--TASK")
parser.add_argument("--GPU", default=0, type=int)
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
FIXED = args.FIXED
LAYER = args.LAYER
TASK = args.TASK
SELECTED_GPU = args.GPU

# LAYER = 6
# SELECTED_GPU = 0
# MODEL_NAME = 'bert'
# FIXED = True
# TASK = "NA"

MAX_LENGTH = 32
BATCH_SIZE = 128
MAX_EPOCHS = 15
LEARNING_RATE = 0.001
LR_SCHEDULER_TYPE = "linear" 
WARMUP_RATIO = 0.0
PATIENCE = 5
SEED = 42
tag = "pretrained_MLM" if FIXED else "finetuned_MLM"
LOAD_MODEL_PATH = f"/home/hmohebbi/Projects/ValueZeroing/directory/models/{MODEL_NAME}/{TASK}/full_forseqclassification_{tag}.pt"
SAVE_MDL_PATH = f"/home/hmohebbi/Projects/blank_out/directory/mdls/{MODEL_NAME}/{TASK}/{tag}/"

# Import Packages
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
from tqdm.auto import tqdm
import json
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from datasets import (
    load_dataset,
    load_from_disk,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdamW,
    get_scheduler,
    default_data_collator,
    set_seed,
)
from modeling.customized_modeling_bert import BertForMaskedLM
# from modeling.customized_modeling_roberta import RobertaForMaskedLM
# from modeling.customized_modeling_electra import ElectraForMaskedLM
set_seed(SEED)

from utils.utils import PREPROCESS_FUNC, MODEL_PATH, NUM_LABELS, BLIMP_TASKS
from diagnostic_model import DiagNet

if not os.path.exists(SAVE_MDL_PATH):
    os.makedirs(SAVE_MDL_PATH)

# GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()

### Load data
if TASK in BLIMP_TASKS:
    data_path = f"/home/hmohebbi/Projects/ValueZeroing/data/processed_blimp/{MODEL_NAME}/{TASK}"
    data = load_from_disk(data_path)["test"]
else:
    print("Not implemented yet!")
    exit()

data = data.shuffle(SEED)
num_labels = NUM_LABELS[TASK]

config = AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME], num_labels=num_labels) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[MODEL_NAME])   

### Preprocessing
dataset = PREPROCESS_FUNC[TASK](data, tokenizer, max_length=MAX_LENGTH, input_masking=True, mlm=False)
num_examples = len(dataset)

# initial
timesteps = [.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
train_examples_count = round(num_examples * timesteps[0]/100)
MDL = train_examples_count * np.log2(num_labels)
mdls = [MDL]

# run
for idx, t in enumerate(timesteps[:-1]):
    # Load Model & Loss & Optimizer
    if MODEL_NAME == "bert":
        base_model = BertForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
    # elif MODEL_NAME == "roberta":
    #     base_model = RobertaForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
    # elif MODEL_NAME == "electra":
    #     base_model = ElectraForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
    if not FIXED:
        base_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
        print("Weights are loaded")
    base_model.to(device)
    base_model.eval()
    diag_model = DiagNet(config)
    diag_model.to(device)

    # Train and Test Datasets
    train_examples_count = round(num_examples * t/100)
    train_dataset = PREPROCESS_FUNC[TASK](data.select(list(range(train_examples_count))), tokenizer, MAX_LENGTH, input_masking=True, mlm=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=BATCH_SIZE)
    train_steps = np.ceil(train_examples_count / BATCH_SIZE)

    test_examples_count = int(np.floor(num_examples * (timesteps[idx + 1] - t)/100))
    test_dataset = PREPROCESS_FUNC[TASK](data.select(list(range(train_examples_count, train_examples_count+test_examples_count))), tokenizer, MAX_LENGTH, input_masking=True, mlm=False)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=BATCH_SIZE)
    test_steps = np.ceil(test_examples_count / BATCH_SIZE)

    print('number of Train examples: ', train_examples_count)
    print('number of Test examples: ', test_examples_count)

    loss_fnc = CrossEntropyLoss() #reduction='sum'
    optimizer = AdamW(diag_model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
            name=LR_SCHEDULER_TYPE,
            optimizer=optimizer,
            num_warmup_steps=WARMUP_RATIO * train_steps,
            num_training_steps=train_steps,
        )

    # Train
    patience_count = 0
    last_test_loss = 1000
    test_loss_per_epoch = []
    diag_model.train()
    for epoch in range(MAX_EPOCHS):
        # Train
        all_test_losses = []
        diag_model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Feature Extraction
            with torch.no_grad():
                outputs = base_model(batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        # token_type_ids=batch['token_type_ids'], 
                        target_index=batch['target_index'] if TASK == "sva" or TASK in BLIMP_TASKS else None,
                        output_hidden_states=True
                        )
            
                hidden_states = torch.stack(outputs['hidden_states'])[LAYER]
                if TASK == "sva" or TASK in BLIMP_TASKS:
                    target_hidden_state = hidden_states[torch.arange(hidden_states.size(0)), batch['target_index']]
                else: # [CLS]
                    target_hidden_state = hidden_states[:, 0]
            
            probs = diag_model(target_hidden_state)
            
            loss = loss_fnc(probs, batch['labels'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Eval
        cumulative_loss = 0
        diag_model.eval()
        for step, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = base_model(batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        # token_type_ids=batch['token_type_ids'], 
                        target_index=batch['target_index'] if TASK == "sva" or TASK in BLIMP_TASKS else None,
                        output_hidden_states=True
                        )

                hidden_states = torch.stack(outputs['hidden_states'])[LAYER]
                if TASK == "sva" or TASK in BLIMP_TASKS:
                    target_hidden_state = hidden_states[torch.arange(hidden_states.size(0)), batch['target_index']]
                else: # [CLS]
                    target_hidden_state = hidden_states[:, 0]

                probs = diag_model(target_hidden_state)
            loss = loss_fnc(probs, batch['labels'])
            all_test_losses.append(loss.item())
            cumulative_loss += loss.item() * batch['input_ids'].shape[0]
        test_loss_per_epoch.append(cumulative_loss)
        
        # Early stopping
        if np.mean(all_test_losses) >= last_test_loss:
            patience_count += 1
            if patience_count >= PATIENCE:
                print('Early stopping! epoch: ', epoch)
                break

        else:
            patience_count = 0
        
        last_test_loss = np.mean(all_test_losses)
    
    MDL += np.min(test_loss_per_epoch) 
    mdls.append(MDL)

        

# Metrics
uniform_codelength = num_examples * np.log2(num_labels)
codelength = round(MDL / 1024, 4)
compression = round(uniform_codelength / MDL, 4)
print(f"Online codelength for layer #{LAYER}: {codelength} kbits")
print(f"Compression for layer #{LAYER}: {compression}") 
print()
print(mdls)

# Save
with open(f'{SAVE_MDL_PATH}codelength_{LAYER}.json', 'w') as f:
    json.dump(codelength, f)
with open(f'{SAVE_MDL_PATH}compression_{LAYER}.json', 'w') as f:
    json.dump(compression, f)





