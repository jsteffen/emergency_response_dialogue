import random

import sys
import string

import nlpaug.augmenter.word as naw
#back_trans_de_en = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-de-en', to_model_name='Helsinki-NLP/opus-mt-en-de', device='cpu')
back_trans_de_fr = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-de-fr', to_model_name='Helsinki-NLP/opus-mt-fr-de', device='cpu')

pronouns = ['dieses', 'dieser', 'diesen', 'diesem', 'das', 'ich', 'mein', 'meine', 'meiner', 'meines', 'meins', 'meinen', 'du', 'dein', 'deine', 'deiner', 'deines', 'ihr', 'ihres', 'ihren', 'ihrer', 'ihrem', 'wir', 'uns', 'unseren', 'unseres', 'sie', 'er', 'sein', 'seine', 'seinen', 'seiner', 'seines', 'es', 'mich', 'dich', 'mir', 'dir', 'uns', 'ihm', 'euch']

def get_backtrans(input_text):
    if input_text.lower() in pronouns:
        return input_text
    aug_text = back_trans_de_fr.augment(input_text)
    return aug_text


def backtranslate_samples(fname):
    new_samples = []
    with open(fname) as f:
        lines = f.readlines()[1:]
        for line in lines:
            splitted = line.split(',')
            text = splitted[2]
            print('Original:', text)
            new_text = get_backtrans(text).replace(',','')
            print('Backtranslated:', new_text)
            print()
            #new_samples.append(line)
            new_line = splitted[0]+','+splitted[1]+','+new_text+','+splitted[3]
            new_samples.append(new_line)
    return new_samples


def write_new_samples(new_fname, new_samples):
    with open(new_fname,'w') as f:
        f.write('id,speakers,tokens,tags\n')
        random.shuffle(new_samples)
        for sample in new_samples:
            f.write(sample)

dtype = 'train' # 'train' or 'dev'
fname = 'csv_da_annotations/csv_low_resource/'+dtype+'.csv'
new_fname = 'csv_da_annotations/csv_low_resource/backtranslated_only_from_fr_'+dtype+'.csv'
new_samples = backtranslate_samples(fname)
write_new_samples(new_fname, new_samples)

