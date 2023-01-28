import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import spacy
import neuralcoref
from transformers import AutoTokenizer

def expand_contraction_process(example):
    example['sentence_good'] = example['sentence_good'].replace("wouldn't", "would not")
    example['sentence_bad'] = example['sentence_bad'].replace("wouldn't", "would not")

    example['sentence_good'] = example['sentence_good'].replace("couldn't", "could not")
    example['sentence_bad'] = example['sentence_bad'].replace("couldn't", "could not")

    example['sentence_good'] = example['sentence_good'].replace("shouldn't", "should not")
    example['sentence_bad'] = example['sentence_bad'].replace("shouldn't", "should not")

    example['sentence_good'] = example['sentence_good'].replace("won't", "will not")
    example['sentence_bad'] = example['sentence_bad'].replace("won't", "will not")

    example['sentence_good'] = example['sentence_good'].replace("can't", "cannot")
    example['sentence_bad'] = example['sentence_bad'].replace("can't", "cannot")

    example['sentence_good'] = example['sentence_good'].replace("don't", "do not")
    example['sentence_bad'] = example['sentence_bad'].replace("don't", "do not")

    example['sentence_good'] = example['sentence_good'].replace("doesn't", "does not")
    example['sentence_bad'] = example['sentence_bad'].replace("doesn't", "does not")

    example['sentence_good'] = example['sentence_good'].replace("didn't", "did not")
    example['sentence_bad'] = example['sentence_bad'].replace("didn't", "did not")

    example['sentence_good'] = example['sentence_good'].replace("isn't", "is not")
    example['sentence_bad'] = example['sentence_bad'].replace("isn't", "is not")

    example['sentence_good'] = example['sentence_good'].replace("aren't", "are not")
    example['sentence_bad'] = example['sentence_bad'].replace("aren't", "are not")

    example['sentence_good'] = example['sentence_good'].replace("wasn't", "was not")
    example['sentence_bad'] = example['sentence_bad'].replace("wasn't", "was not")

    example['sentence_good'] = example['sentence_good'].replace("weren't", "were not")
    example['sentence_bad'] = example['sentence_bad'].replace("weren't", "were not")

    example['sentence_good'] = example['sentence_good'].replace("hasn't", "has not")
    example['sentence_bad'] = example['sentence_bad'].replace("hasn't", "has not")

    example['sentence_good'] = example['sentence_good'].replace("haven't", "have not")
    example['sentence_bad'] = example['sentence_bad'].replace("haven't", "have not")

    example['sentence_good'] = example['sentence_good'].replace("hadn't", "had not")
    example['sentence_bad'] = example['sentence_bad'].replace("hadn't", "had not")

    return example

def sva_process(raw_data):
    data = {}
    data['sentence_good'] = []
    data['sentence_bad'] = []
    data['good_word'] = []
    data['bad_word'] = []
    data['target_index'] = []
    data['cue_indices'] = []
    data['labels'] = []
    for example in raw_data:
        doc_good = nlp(example['sentence_good'])
        doc_bad = nlp(example['sentence_bad'])
        cue_index = -1
        sentence_good = []
        sentence_bad = []
        for i in range(len(doc_good)):
            sentence_good.append(doc_good[i].text)
            sentence_bad.append(doc_bad[i].text)
            if doc_good[i].dep_ == "nsubj" and cue_index == -1:
                cue_index = i
            if doc_good[i].text != doc_bad[i].text:
                target_index = i
                good_word = doc_good[i].text
                bad_word = doc_bad[i].text

                tag = doc_good[target_index].tag_
                if tag == 'VBZ':
                    tag = 'singular'
                elif tag == 'VBP':
                    tag = 'plural'

                # fix wrong tags of SpaCy
                elif tag == "NN": # it's vice versa becuse spacy consideres verb as a noun, so those plural nouns are actually a singular verb e.g., works
                    tag = "plural"
                elif tag == "NNS":
                    tag = "singular"
                elif doc_good[target_index].tag_ not in ['VBZ', 'VBP'] and doc_bad[target_index].tag_ in ['VBZ', 'VBP']:
                    if doc_bad[target_index].tag_ == "VBZ":
                        tag = "plural" 
                    elif doc_bad[target_index].tag_ == "VBP":
                        tag = "singular" 
                
                elif tag not in ['VBZ', 'VBP']: #correcting exceptions
                    if doc_good[target_index].text in ["was", "upsets", "hurts", "bores", "vanishes", "distracts", "kisses", "boycotts", "scares"]:
                        tag = "singular" 
                    elif doc_good[target_index].text in ["were", "upset", "hurt", "bore", "vanish", "distract", "kiss", "boycott", "scare"]:
                        tag = "plural" 
                    else:
                        print(doc_good[target_index].text, doc_bad[target_index].text)
    
        
        if target_index == -1 or cue_index == -1:
            continue
        if len(tokenizer_bert.tokenize(good_word)) > 1 or len(tokenizer_bert.tokenize(bad_word)) > 1:
            continue
            
        data['sentence_good'].append(sentence_good)
        data['sentence_bad'].append(sentence_bad)
        data['target_index'].append(target_index)
        data['cue_indices'].append([cue_index])
        data['good_word'].append(good_word)
        data['bad_word'].append(bad_word)
        data['labels'].append(tag)

    return Dataset.from_dict(data)

def det_process(raw_data):
    data = {}
    data['sentence_good'] = []
    data['sentence_bad'] = []
    data['good_word'] = []
    data['bad_word'] = []
    data['target_index'] = []
    data['cue_indices'] = []
    data['labels'] = []
    for ex, example in enumerate(raw_data):
        doc_good = nlp(example['sentence_good'])
        doc_bad = nlp(example['sentence_bad'])
        edges = []
        for w in doc_good:
            edges.extend([(w.i, child.i) for child in w.children])

        target_index = -1
        cue_index = -1
        sentence_good = []
        sentence_bad = []
        for i in range(len(doc_good)):
            sentence_good.append(doc_good[i].text)
            sentence_bad.append(doc_bad[i].text)
            if doc_good[i].text != doc_bad[i].text: # doc_good[i].dep_ == "det" and
                target_index = i
                good_word = doc_good[i].text
                bad_word = doc_bad[i].text

            for s, d in edges:
                if d == target_index:
                    cue_index = s
                    break

        if target_index == -1 or cue_index == -1:
            print(ex)
            continue
        if len(tokenizer_bert.tokenize(good_word)) > 1 or len(tokenizer_bert.tokenize(bad_word)) > 1:
            continue
        
        if good_word in ['this', 'that']:
            tag = "singular"
        elif good_word in ['these', 'those']:
            tag = "plural"
        else:
            print(good_word)

        data['sentence_good'].append(sentence_good)
        data['sentence_bad'].append(sentence_bad)
        data['target_index'].append(target_index)
        data['cue_indices'].append([cue_index])
        data['good_word'].append(good_word)
        data['bad_word'].append(bad_word)
        data['labels'].append(tag)

    return Dataset.from_dict(data)

def number_process(raw_data):
    data = {}
    data['sentence_good'] = []
    data['sentence_bad'] = []
    data['good_word'] = []
    data['bad_word'] = []
    data['target_index'] = []
    data['cue_indices'] = []
    data['labels'] = []
    for ex, example in enumerate(raw_data):
        doc_good = nlp(example['sentence_good'])
        doc_bad = nlp(example['sentence_bad'])
        if not doc_good._.has_coref:
            continue
        cue_words, _ = doc_good._.coref_clusters[0]
        cue_words = cue_words.text.split(" ")
        target_index = -1
        cue_indices = []
        sentence_good = []
        sentence_bad = []
        for i in range(len(doc_good)):
            sentence_good.append(doc_good[i].text)
            sentence_bad.append(doc_bad[i].text)
            if doc_good[i].text != doc_bad[i].text: 
                target_index = i
                good_word = doc_good[i].text
                bad_word = doc_bad[i].text

            if doc_good[i].text in cue_words:
                cue_indices.append(i)

        if target_index == -1 or not cue_indices:
            continue
        if len(tokenizer_bert.tokenize(good_word)) > 1 or len(tokenizer_bert.tokenize(bad_word)) > 1:
            continue

        if good_word in ['itself', 'himself', 'herself']:
            tag = "singular"
        elif good_word in ['themselves']:
            tag = "plural"
        else:
            print(good_word)
        
        data['sentence_good'].append(sentence_good)
        data['sentence_bad'].append(sentence_bad)
        data['target_index'].append(target_index)
        data['cue_indices'].append(cue_indices)
        data['good_word'].append(good_word)
        data['bad_word'].append(bad_word)
        data['labels'].append(tag)

    return Dataset.from_dict(data)


TASK_UID = {
    'anaphor_number_agreement': 'ana',
    'determiner_noun_agreement_2': 'dna',
    'determiner_noun_agreement_with_adj_2': 'dnaa',
    'distractor_agreement_relational_noun': 'darn',
    'regular_plural_subject_verb_agreement_1': 'rpsv',
}

UID_PROCESSOR = {
    'ana': number_process,
    'dna': det_process,
    'dnaa': det_process,
    'rpsv': sva_process,
    'darn': sva_process,
}

SEED = 12
# Load Tokenizer 
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased") 

# Load spacy
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

for task, uid in TASK_UID.items():
    raw_data = load_dataset("blimp", task)['train']
    raw_data = raw_data.map(expand_contraction_process)

    data = UID_PROCESSOR[uid](raw_data)
    data = data.shuffle(seed=SEED)

    if uid == "rpsv": # balancing class labels
        plur_indices = np.where(np.array(data['labels']) == 'plural')[0]
        sing_indices = np.where(np.array(data['labels']) == 'singular')[0]
        sing_indices = np.random.choice(sing_indices, len(plur_indices))
        plur_data = data.select(plur_indices)
        sing_data = data.select(sing_indices)
        data = concatenate_datasets([plur_data, sing_data])
        data = data.shuffle(seed=SEED)

    # aggregate datasets
    if uid == "ana":
        number_dataset = data
    else:
        number_dataset = concatenate_datasets([number_dataset, data])
        number_dataset = number_dataset.shuffle(seed=SEED)
    
number_dataset = number_dataset.shuffle(seed=SEED)
number_dataset = number_dataset.train_test_split(test_size=0.5)
number_dataset.save_to_disk(f"/home/hmohebbi/Projects/ValueZeroing/data/processed_blimp/bert/NA")

