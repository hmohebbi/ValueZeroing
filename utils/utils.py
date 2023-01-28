import numpy as np

NUM_LABELS = {
    "ana": 2,
    "dna": 2,
    "dnaa": 2,
    "rpsv": 2,
    "darn": 2,
    "NA": 2,
}

blimp_to_label = {
    'singular': 0,
    'plural': 1,
}

MODEL_PATH = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'electra': 'google/electra-base-generator',
}


BLIMP_TASKS = [
    "ana",
    'dna',
    "dnaa",
    "rpsv",
    "darn",
    "NA",
]

def blimp_to_features(data, tokenizer, max_length, input_masking, mlm):
    all_features = []
    for example in data:
        text = example['sentence_good']
        tokens = []
        cue_indices = []
        # token to id
        for w_ind, word in enumerate(text):
            ids = tokenizer.encode(word, add_special_tokens=False)
            if w_ind in example['cue_indices']:
                cue_indices.append(len(tokens))
            if w_ind == example['target_index']:
                target_index = len(tokens)
            tokens.extend(ids)
        

        tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        cue_indices = [x+1 for x in cue_indices] # 'cause of adding cls
        target_index += 1 # 'cause of adding cls
        if input_masking:
            tokens[target_index] = tokenizer.mask_token_id

        # padding
        length = len(tokens)
        inputs = {}
        inputs['input_ids'] = tokens if max_length is None else tokens + [tokenizer.pad_token_id]*(max_length - length)
        inputs['attention_mask'] = [1]*length if max_length is None else [1]*length + [0]*(max_length - length)
        # inputs['token_type_ids'] = [0]*length if max_length is None else [0]*max_length
        inputs['target_index'] = target_index
        inputs['labels'] = tokenizer.convert_tokens_to_ids(example['good_word']) if mlm else blimp_to_label[example['labels']]
        inputs['good_token_id'] = tokenizer.convert_tokens_to_ids(example['good_word'])
        inputs['bad_token_id'] = tokenizer.convert_tokens_to_ids(example['bad_word'])
        if max_length is None:
            inputs['cue_indices'] = cue_indices

        all_features.append(inputs)
    return all_features[0] if len(all_features) == 1 else all_features


PREPROCESS_FUNC = {
    'ana': blimp_to_features,
    'dna': blimp_to_features,
    'dnaa': blimp_to_features,
    'rpsv': blimp_to_features,
    'darn': blimp_to_features,
    'NA': blimp_to_features,
}