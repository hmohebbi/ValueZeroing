import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--TASK")
parser.add_argument("--SPLIT")
parser.add_argument("--MODEL_NAME")
parser.add_argument("--FIXED", action="store_true")
parser.add_argument("--CHECKPOINT")
parser.add_argument("--METRIC")
parser.add_argument("--INPUT_MASKING", action="store_true")
parser.add_argument("--MLM", action="store_true")
parser.add_argument("--SAVE_SCORES", action="store_true")
parser.add_argument("--GPU", default=0, type=int)
args = parser.parse_args()

TASK = args.TASK
SPLIT = args.SPLIT
MODEL_NAME = args.MODEL_NAME
FIXED = args.FIXED
CHECKPOINT = args.CHECKPOINT
METRIC = args.METRIC
INPUT_MASKING = args.INPUT_MASKING
MLM = args.MLM
SAVE_SCORES = args.SAVE_SCORES
SELECTED_GPU = args.GPU

# SELECTED_GPU = 0
# MODEL_NAME = 'bert'
# FIXED = False # True for pre-trained model and False for finetuned model
# CHECKPOINT = "full" 
# METRIC = 'cosine' 
# TASK = "NA"
# SPLIT = "test"
# INPUT_MASKING = True
# MLM = True
# SAVE_SCORES = False

SEED = 42
tag = "pretrained" if FIXED else "finetuned"
if MLM:
    tag += "_MLM"
masking_tag = "masked" if INPUT_MASKING else "full"
LOAD_MODEL_PATH = f"/home/hmohebbi/Projects/ValueZeroing/directory/models/{MODEL_NAME}/{TASK}/{CHECKPOINT}_forseqclassification_{tag}.pt"
SAVE_SCORES_PATH = f"/home/hmohebbi/Projects/ValueZeroing/directory/scores/{MODEL_NAME}/{TASK}/{tag}/{masking_tag}/"


### Import Libraries
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
from tqdm.auto import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_distances

import torch
from torch.utils.data import DataLoader

from utils.utils import PREPROCESS_FUNC, MODEL_PATH, NUM_LABELS, BLIMP_TASKS

from datasets import (
    load_from_disk,
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from modeling.customized_modeling_bert import BertForSequenceClassification, BertForMaskedLM
set_seed(SEED)


def compute_joint_attention(att_mat, res=True):
    if res:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        att_mat = att_mat + residual_att
        att_mat = att_mat / att_mat.sum(axis=-1)[...,None]
    
    joint_attentions = np.zeros(att_mat.shape)
    layers = joint_attentions.shape[0]
    joint_attentions[0] = att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions

DISTANCE_FUNC = {'cosine': cosine_distances}

if SAVE_SCORES and not os.path.exists(SAVE_SCORES_PATH):
    os.makedirs(SAVE_SCORES_PATH)

### GPU
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
    data = load_from_disk(data_path)[SPLIT]
else:
    print("Not implemented yet!")
    exit()
data = data.shuffle(SEED)
num_labels = NUM_LABELS[TASK]

### Load Tokenizer & Model
config = AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME], num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[MODEL_NAME])   
if MLM:
    model = BertForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
else:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH[MODEL_NAME], config=config, target_mode="mask" if TASK == "sva" or TASK in BLIMP_TASKS else "cls")
### load weights for fine-tuned model
if not (MLM and FIXED):
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    print("Weights are loaded")
model.to(device)
model.eval()

### Preprocessing
dataset = PREPROCESS_FUNC[TASK](data, tokenizer, max_length=None, input_masking=INPUT_MASKING, mlm=MLM)
num_examples = len(dataset)
dataloader = DataLoader(dataset, collate_fn= default_data_collator, batch_size=1)

### run
# # layerwise, using zero value
all_valuezeroing_scores = [] # (#layers, #seq_length, #seq_length)
all_rollout_valuezeroing_scores = [] # (#layers, #seq_length, #seq_length)

progress_bar = tqdm(range(num_examples))
for step, inputs in enumerate(dataloader):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        # token_type_ids=inputs['token_type_ids'], 
                        output_hidden_states=True, output_attentions=False)

    org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
    input_shape = inputs['input_ids'].size() 
    batch_size, seq_length = input_shape

    ## layerwise zeroing value
    score_matrix = np.zeros((config.num_hidden_layers, seq_length, seq_length))
    for l, layer_module in enumerate(model.bert.encoder.layer):
        for t in range(seq_length):
            extended_blanking_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(inputs['attention_mask'], input_shape, device)
            with torch.no_grad():
                layer_outputs = layer_module(org_hidden_states[l].unsqueeze(0), # previous layer's original output 
                                            attention_mask=extended_blanking_attention_mask,
                                            output_attentions=False,
                                            zero_value_index=t,
                                            )
            hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
            # compute similarity between original and new outputs
            # cosine
            x = hidden_states
            y = org_hidden_states[l+1].detach().cpu().numpy()
            
            distances = DISTANCE_FUNC[METRIC](x, y).diagonal()
            score_matrix[l, :, t] = distances
            
              
    score_matrix = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)
    
    all_valuezeroing_scores.append(score_matrix)
    all_rollout_valuezeroing_scores.append(compute_joint_attention(score_matrix, res=False))
    
    progress_bar.update(1)


### Save scores
if SAVE_SCORES:
    # Value Zeroing
    with open(f'{SAVE_SCORES_PATH}{CHECKPOINT}_{METRIC.capitalize()}_valuezeroing.pkl', 'wb') as f:
        pickle.dump(all_valuezeroing_scores, f)
    # Value Zeroing + rollout
    with open(f'{SAVE_SCORES_PATH}{CHECKPOINT}_rollout_{METRIC.capitalize()}_valuezeroing.pkl', 'wb') as f:
        pickle.dump(all_rollout_valuezeroing_scores, f)
