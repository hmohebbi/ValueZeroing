
SELECTED_GPU = 0
MODEL_NAME = 'bert'
FIXED = False # True for pre-trained model and False for finetuned model
METRIC = 'cosine' 
TASK = "NA"
LOAD_MODEL_PATH = f"directory/models/{MODEL_NAME}/{TASK}/full_forseqclassification_finetuned_MLM.pt"


### Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

import torch


from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from modeling.customized_modeling_bert import BertForMaskedLM

## Rollout Helper Function
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

DISTANCE_FUNC = {
    'cosine': cosine_distances
}
MODEL_PATH = {
    'bert': 'bert-base-uncased',
}


## GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()


## Load Tokenizer & Model
config = AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME])
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[MODEL_NAME])   
model = BertForMaskedLM.from_pretrained(MODEL_PATH[MODEL_NAME], config=config)
### load weights for fine-tuned model
if not FIXED:
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    print("Weights are loaded")
model.to(device)
model.eval()

## Preprocessing
text = "the pictures of some hat [MASK] scaring marcus."
inputs = tokenizer(text, return_tensors="pt")

## Cpmpute: layerwise value zeroing
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'], 
                    output_hidden_states=True, output_attentions=False)

org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
input_shape = inputs['input_ids'].size() 
batch_size, seq_length = input_shape

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
        
            
valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)
rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, res=False)


# Plot:
cmap = "Blues"
all_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]
LAYERS = list(range(12))
fig, axs = plt.subplots(3, 4, figsize=(30, 20))
plt.subplots_adjust(hspace = 0.5, wspace=0.5)
for layer in LAYERS:
    a = (layer)//4
    b = layer%4
    sns.heatmap(ax=axs[a, b], data=pd.DataFrame(rollout_valuezeroing_scores[layer], index= all_tokens, columns=all_tokens), cmap=cmap, annot=False, cbar=False)
    axs[a, b].set_title(f"Layer: {layer+1}")