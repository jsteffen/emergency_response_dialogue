import random
from copy import deepcopy
import sys
import string
from transformers import BertTokenizer, BertForMaskedLM
import torch

model_name = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)    
mask_model = BertForMaskedLM.from_pretrained(model_name)

pronouns = ['dieses', 'dieser', 'diesen', 'diesem', 'das', 'ich', 'mein', 'meine', 'meiner', 'meines', 'meins', 'meinen', 'du', 'dein', 'deine', 'deiner', 'deines', 'ihr', 'ihres', 'ihren', 'ihrer', 'ihrem', 'wir', 'uns', 'unseren', 'unseres', 'sie', 'er', 'sein', 'seine', 'seinen', 'seiner', 'seines', 'es', 'mich', 'dich', 'mir', 'dir', 'uns', 'ihm', 'euch']

def get_masked_augmentation(input_text, proportion):
    aug_text = ''
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    num_replacements = int(proportion*len(input_tokens))
    # we don't replace very short, e.g., one-word sentences
    if num_replacements==0:
        return input_text
    index_range = [i for i in range(len(input_ids))]
    masked_indices = random.choices(index_range, k=num_replacements)
    masked_indices.sort()
    masked_ids = deepcopy(input_ids)
    for masked_index in masked_indices:
        try:
            masked_ids[masked_index] = tokenizer.mask_token_id
        except:
            print('masked_indices:', masked_indices, '\n', 'masked_ids:', masked_ids, '\n', 'masked_index:', masked_index)
            sys.exit(5)
    orig_ids = []
    for idx in masked_indices:
        orig_ids.append(input_ids[idx])
    result = mask_model(torch.tensor([masked_ids]))
    grouped_ids = result[0][:, masked_indices].topk(10).indices.tolist()[0]
    for group_i, pred_ids in enumerate(grouped_ids):
        masked_index = masked_indices[group_i]
        for pred_id in pred_ids:
            masked_ids[masked_index] = pred_id
            predicted_word = tokenizer.decode([pred_id]).replace(' ','')#.lower()
            print('PREDICTED:', group_i, masked_index, predicted_word, 'vs', tokenizer.decode([input_ids[masked_index]]))
            if pred_id!=orig_ids[group_i] and not(predicted_word.startswith('##')) and not('unused_punctuation' in predicted_word): # and not(predicted_word in string.punctuation)
                print('>>>>>', predicted_word)
                break
    print('Original:', tokenizer.decode(input_ids))
    print('Augmented:', tokenizer.decode(masked_ids))
    aug_text = tokenizer.decode(masked_ids)
    return aug_text

def get_random_augmentation(input_text, proportion):
    aug_text = ''
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    num_replacements = int(proportion*len(input_tokens))
    # we don't replace very short, e.g., one-word sentences
    if num_replacements==0:
        return input_text
    index_range = [i for i in range(len(input_ids))]
    modified_indices = random.choices(index_range, k=num_replacements)
    operations = random.choices(['delete','insert','swap'], k=num_replacements)
    idx2operation = dict()
    for i, idx in enumerate(modified_indices):
        idx2operation[idx] = operations[i]
    assert(len(modified_indices)==len(operations))
    new_subtokens = deepcopy(input_ids)
    # idx is a position
    for idx, _ in enumerate(input_ids):
        if idx in idx2operation:
            op = idx2operation[idx]
            if op=='delete':
                new_subtokens[idx] = None
            # if we use insert we create a list of subtokens
            elif op=='insert':
                # pick the new subtoken from the list of available ones in the doc
                new_subtoken = random.choice(input_ids) # w/o CLS and SEP
                #assert(not(new_subtoken in ['[CLS]', '[SEP]']))
                if type(new_subtokens[idx]) is list:
                      new_subtokens[idx].append(new_subtoken)
                else:
                      new_subtokens[idx] = [new_subtokens[idx], new_subtoken]
            elif op=='swap':
                other_idx = random.choice(index_range)
                other_value = input_ids[other_idx]
                #assert(not(other_value in ['[CLS]', '[SEP]']))
                #assert(not(new_subtokens[idx] in ['[CLS]', '[SEP]']))
                new_subtokens[other_idx] = new_subtokens[idx]
                new_subtokens[idx] = other_value
    edited_subtokens = []
    for new_id, el in enumerate(new_subtokens):
        if type(el) is list:
            edited_subtokens.extend(el)
        elif el is not None:
            edited_subtokens.append(el)
    print('Original:', tokenizer.decode(input_ids))
    print('Augmented:', tokenizer.decode(edited_subtokens))
    aug_text = tokenizer.decode(edited_subtokens)
    return aug_text

def augment_samples(fname, aug_type, num_of_rounds, proportion):
    new_samples = []
    with open(fname) as f:
        lines = f.readlines()[1:]
        for line in lines:
            splitted = line.split(',')
            text = splitted[2]
            new_samples.append(line) # adding the original text
            for k in range(num_of_rounds):
                if aug_type=='masked':
                    new_text = get_masked_augmentation(text, proportion).replace(',','')
                elif aug_type=='random':
                    new_text = get_random_augmentation(text, proportion).replace(',','')
                else:
                    raise Exception('Unknown augmentation type:', aug_type)
                #print('Augmented:', new_text)
                new_line = splitted[0]+','+splitted[1]+','+new_text+','+splitted[3]
                new_samples.append(new_line)
    return new_samples


def write_new_samples(new_fname, new_samples):
    with open(new_fname,'w') as f:
        f.write('id,speakers,tokens,tags\n')
        random.shuffle(new_samples)
        for sample in new_samples:
            f.write(sample)


aug_type = 'masked' # 'masked' or 'random'
#dtype = 'dev' # 'train' or 'dev'
#proportion = 0.2 # 0.1 0.2 0.4 0.6
#num_of_rounds = 5 # 1 2 5 10

for dtype in ['dev', 'train']:
    for proportion in [0.1, 0.2, 0.4, 0.6]:
        for num_of_rounds in [1, 2, 5, 10]:
            setting = aug_type+'_'+str(proportion)+'_'+str(num_of_rounds)
            print('Setting:', setting)
            fname = 'csv_da_annotations/csv_low_resource/'+dtype+'.csv'
            new_fname = 'csv_da_annotations/csv_low_resource/'+setting+'_'+dtype+'.csv'    
            new_samples = augment_samples(fname, aug_type, num_of_rounds, proportion)
            write_new_samples(new_fname, new_samples)

