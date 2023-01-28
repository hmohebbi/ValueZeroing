import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--FIXED", action="store_true")
parser.add_argument("--TASK")
parser.add_argument("--MAX_LENGTH", type=int)
parser.add_argument("--BATCH_SIZE", type=int)
parser.add_argument("--EPOCHS", type=int)
parser.add_argument("--GPU", default=0, type=int)
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
FIXED = args.FIXED
TASK = args.TASK
NUM_TRAIN_EPOCHS = args.EPOCHS
MAX_LENGTH = args.MAX_LENGTH
PER_DEVICE_BATCH_SIZE = args.BATCH_SIZE
SELECTED_GPU = args.GPU

# SELECTED_GPU = 0
# MODEL_NAME = 'bert'
# FIXED = False
# TASK = "NA"
# MAX_LENGTH = 32
# NUM_TRAIN_EPOCHS = 5
# PER_DEVICE_BATCH_SIZE = 64

INPUT_MASKING = True
MLM = True
LEARNING_RATE = 3e-5
LR_SCHEDULER_TYPE = "linear" 
WARMUP_RATIO = 0.1
SEED = 42
SAVED_MODEL_PATH = f"/home/hmohebbi/Projects/ValueZeroing/directory/models/{MODEL_NAME}/{TASK}/"

# Import Packages
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from utils.utils import PREPROCESS_FUNC, MODEL_PATH, NUM_LABELS, BLIMP_TASKS

from datasets import (
    load_dataset, 
    load_from_disk,
    load_metric,
)
from modeling.customized_modeling_bert import BertForMaskedLM
# from modeling.customized_modeling_roberta import RobertaForMaskedLM
# from modeling.customized_modeling_electra import ElectraForMaskedLM
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdamW,
    get_scheduler,
    default_data_collator,
    set_seed,
)
set_seed(SEED)

if not os.path.exists(SAVED_MODEL_PATH):
    os.makedirs(SAVED_MODEL_PATH)

# GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()

# Load Dataset
if TASK in BLIMP_TASKS:
    data_path = f"/home/hmohebbi/Projects/ValueZeroing/data/processed_blimp/{MODEL_NAME}/{TASK}"
    data = load_from_disk(data_path)
    train_data = data['train']
    eval_data = data['test']
else:
    print("Not implemented yet!")
    exit()
train_data = train_data.shuffle(SEED)
num_labels = NUM_LABELS[TASK]

# Download Tokenizer & Model
config = AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME], num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[MODEL_NAME])  

if MODEL_NAME == "bert":
    model = BertForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
# elif MODEL_NAME == "roberta":
#     model = RobertaForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
# elif MODEL_NAME == "electra":
#     model = ElectraForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
else:
    print("model doesn't exist")
    exit()

model.to(device)

# Preprocessing
train_dataset = PREPROCESS_FUNC[TASK](train_data, tokenizer, MAX_LENGTH, input_masking=INPUT_MASKING, mlm=MLM)
eval_dataset = PREPROCESS_FUNC[TASK](eval_data, tokenizer, MAX_LENGTH, input_masking=INPUT_MASKING, mlm=MLM)

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn= default_data_collator, batch_size=PER_DEVICE_BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, collate_fn= default_data_collator, batch_size=PER_DEVICE_BATCH_SIZE)

num_update_steps_per_epoch = len(train_dataloader)
max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch 

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_RATIO * max_train_steps,
        num_training_steps=max_train_steps,
    )

# metric & Loss
metric = load_metric("accuracy")
loss_fct = CrossEntropyLoss()

tag = "forseqclassification_"
tag += "pretrained" if FIXED else "finetuned" 
if MLM:
    tag += "_MLM"

# Train
progress_bar = tqdm(range(max_train_steps))
completed_steps = 0
for epoch in range(NUM_TRAIN_EPOCHS):
    # Train
    model.train()
    for batch in train_dataloader:
        good_token_id = batch.pop('good_token_id').to(device)
        bad_token_id = batch.pop('bad_token_id').to(device)
        batch = {k: v.to(device) for k, v in batch.items()} 
        outputs = model(**batch)
        logits = outputs.logits
        
        good_logits = logits[torch.arange(logits.size(0)), good_token_id]
        bad_logits = logits[torch.arange(logits.size(0)), bad_token_id]
        logits_of_interest = torch.stack([good_logits, bad_logits], dim=1)
        labels = torch.zeros(logits_of_interest.shape[0], dtype=torch.int64, device=device)
        loss = loss_fct(logits_of_interest, labels)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1
    

    model.eval()
    for batch in eval_dataloader:
        if MLM:
            good_token_id = batch.pop('good_token_id').to(device)
            bad_token_id = batch.pop('bad_token_id').to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits

        if MLM:
            good_logits = logits[torch.arange(logits.size(0)), good_token_id]
            bad_logits = logits[torch.arange(logits.size(0)), bad_token_id]
            logits_of_interest = torch.stack([good_logits, bad_logits], dim=1)
            labels = torch.zeros(logits_of_interest.shape[0], dtype=torch.int64, device=device)
            predictions = torch.argmax(logits_of_interest, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        else:
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

    eval_metric = metric.compute()
    print(f"epoch {epoch}: {eval_metric}")    


# Save
torch.save(model.state_dict(), f'{SAVED_MODEL_PATH}full_{tag}.pt')
