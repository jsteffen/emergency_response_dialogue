import datasets
from datasets import load_dataset
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, AutoConfig
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
import torch.nn.init as init
import copy
import sys
import os


import spacy
from spacy.tokens import Doc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WhitespaceTokenizer(object):
        def __init__(self, vocab):
                self.vocab = vocab

        def __call__(self, text):
                words = text.split(' ')
                spaces = [True] * len(words)
                return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("de_core_news_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

pos_tags = {'Noun':0, 'Verb':1, 'Prep':2, 'Pron':3, 'Adv':4, 'Adj':5, 'Other':6, 'O':7}
id2pos_tags = dict()
for k,v in pos_tags.items():
    id2pos_tags[v] = k
pos_emb_size = 300

os.environ["WANDB_DISABLED"] = "true"

max_len_bio = 128
batch_size = 16

do_train = False

anno_type = 'neg_samples' # 'neg_samples'
task = "mittel"# 'auftrag', 'einheit', 'mittel', 'ziel', 'weg'
labels = ['O', 'B', 'I']#, '[PAD]']
id2label = {id_: label for id_, label in enumerate(labels)}
label2id = {label: id_ for id_, label in enumerate(labels)}

model_name = "bert-base-german-cased"
config = AutoConfig.from_pretrained(model_name, num_label=len(labels), id2label=id2label, label2id=label2id)
model = AutoAdapterModel.from_pretrained(model_name)
model.add_adapter(task)

from transformers.adapters.heads import PredictionHead

class PosHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        **config
    ):
        # initialization of the custom head
        super().__init__(head_name)
        self.config = config
        self.build(model=model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.downsize_layer = nn.Linear(768, 300).to(self.device)
        self.hidden_layer = nn.Linear(300+pos_emb_size, 300).to(self.device)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc_layer = nn.Linear(300, len(labels)).to(self.device)
        
    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, pos_input=None, **kwargs):
        #pos_input = torch.transpose(pos_input,0,1)
        outputs = self.downsize_layer(outputs[0])
        outputs = torch.cat((outputs, pos_input), dim=-1)
        outputs = self.hidden_layer(outputs)
        outputs = self.tanh(outputs)
        outputs = self.fc_layer(outputs)
        return outputs

model.register_custom_head("pos_custom_head", PosHead)
config = {"num_labels": len(labels), "layers": 2, "activation_function": "tanh"}
model.add_custom_head(head_type="pos_custom_head", head_name=task+"_head", **config)
tokenizer = AutoTokenizer.from_pretrained(model_name)


bert_model = BertModel.from_pretrained(model_name)

pos_embedder = nn.Embedding(len(pos_tags), pos_emb_size)
init.normal_(pos_embedder.weight, std=0.02)

def get_pos_embeds(tokens):
    tags = [pos_tags['O']]
    for token in nlp(tokens):
        pos_tag = token.tag_
        if pos_tag=='PPER':
            pos_tag = pos_tags['Pron']
        elif pos_tag.startswith('V'):
            pos_tag = pos_tags['Verb']
        elif pos_tag.startswith('N'): # NN NE
            pos_tag = pos_tags['Noun']
        elif pos_tag.startswith('ADV'):
            pos_tag = pos_tags['Adv']
        elif pos_tag.startswith('ADJ'):
            pos_tag = pos_tags['Adj']
        elif pos_tag.startswith('APPR'):
            pos_tag = pos_tags['Prep']
        else:
            pos_tag = pos_tags['Other']

        for subtoken in tokenizer.tokenize(token.text):
            tags.append(pos_tag)
    tags = tags[:max_len_bio-1]
    while len(tags)<(max_len_bio-1):
        tags.append(pos_tags['O'])
    tags = tags+[pos_tags['O']]
    return tags

def encode_data(data):
    encoded = tokenizer([' '.join(doc.split()) for doc in data["tokens"]], pad_to_max_length=True, padding="max_length", max_length=max_len_bio, truncation=True, add_special_tokens=True)
    bert_out = bert_model(torch.tensor(encoded['input_ids']), torch.tensor(encoded['attention_mask']))
    pos_tags = []
    for sample in data['tokens']:
        sample = ' '.join(sample.split())
        sample_pos_tags = get_pos_embeds(sample)
        pos_tags.append(sample_pos_tags)        
    embedded_pos_tags = pos_embedder(torch.tensor(pos_tags)).tolist()
    encoded["pos_input"] = embedded_pos_tags
    encoded["pos_labels"] = pos_tags
    return encoded    
    
def encode_labels(example):
    r_tags = []
    count = 0
    all_tokenized = []
    for idx, token in enumerate(example['tokens'].split()):
        tokenized = tokenizer.tokenize(token)
        all_tokenized.extend(tokenized)
        label = example['tags'].split()[idx]
        for ti, t in enumerate(tokenized):
            if ti!=0:
                if label=='O':
                    r_tags.append(label2id[label])
                else:
                    r_tags.append(label2id['I'])
            else:
                r_tags.append(label2id[label])
    r_tags = [label2id['O']]+r_tags[:max_len_bio-1]+[label2id['O']]
    rest = max_len_bio-len(r_tags)
    if rest>0:
        for i in range(rest):
            r_tags.append(label2id['O'])
    labels = dict()
    labels['labels'] = torch.tensor(r_tags)
    return labels



train_task_dataset = datasets.Dataset.from_csv(anno_type+'_csv/'+anno_type+'_'+task+'_train.csv')
train_task_dataset = train_task_dataset.map(encode_labels)
train_task_dataset = train_task_dataset.map(encode_data, batched=True, batch_size=batch_size)

dev_task_dataset = datasets.Dataset.from_csv(anno_type+'_csv/'+anno_type+'_'+task+'_dev.csv')
dev_task_dataset = dev_task_dataset.map(encode_labels)
dev_task_dataset = dev_task_dataset.map(encode_data, batched=True, batch_size=batch_size)

test_task_dataset = datasets.Dataset.from_csv(anno_type+'_csv/'+anno_type+'_'+task+'_test.csv')
test_task_dataset = test_task_dataset.map(encode_labels)
test_task_dataset = test_task_dataset.map(encode_data, batched=True, batch_size=batch_size)


train_task_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pos_input'])
dev_task_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pos_input'])
test_task_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pos_input'])

print('train:', len(train_task_dataset))
print('dev:', len(dev_task_dataset))
print('test:', len(test_task_dataset))

dataloader = torch.utils.data.DataLoader(train_task_dataset, shuffle=True)
evaluate_dataloader = torch.utils.data.DataLoader(dev_task_dataset)
test_dataloader = torch.utils.data.DataLoader(test_task_dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

if do_train:
    model.set_active_adapters([[task]])
    model.train_adapter([task])

    class_weights = torch.FloatTensor([1, 1.5, 1.5]).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-4,}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=1e-3)

    prev_smallest_dev_loss = None
    best_epoch = None

    print(model)

    for epoch in range(12):
        for i, batch in enumerate(dataloader):
            for k, v in batch.items():
                if k=='pos_input' and len(v)>1:
                    batch[k] = torch.stack(v).to(device)
                else:
                    batch[k] = v.to(device)

            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], pos_input=batch["pos_input"], adapter_names=[task])
            predictions = outputs[0]
            expected = torch.flatten(batch["labels"].long(), 0, 1)
            loss = loss_function(predictions, expected)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10000 == 0:
                print(epoch)
                print(f"loss: {loss}")
        with torch.no_grad():
            predictions_list = []
            expected_list = []
            dev_losses = []
            for i, batch in enumerate(evaluate_dataloader):
                for k, v in batch.items():
                    if k=='pos_input' and len(v)>1:
                        batch[k] = torch.stack(v).to(device)
                    else:
                        batch[k] = v.to(device)
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], pos_input=batch["pos_input"], adapter_names=[task])
                predictions = torch.argmax(outputs[0], 1)
                expected = batch["labels"].float()

                mpredictions = outputs[0]
                mexpected = torch.flatten(batch["labels"].long(), 0, 1)
                loss = loss_function(mpredictions, mexpected)
                dev_losses.append(loss.item())
                predictions_list.append(predictions)
                expected_list.append(expected)
            cur_epoch_dev_loss = round(sum(dev_losses)/len(dev_losses),3)
            print(epoch, 'Dev loss:', cur_epoch_dev_loss)
            # always save the last epoch if there are too few samples in dev data!
            if True:
                # save adapter and head
                model.save_adapter('adapters_pos/'+task+'_adapter/', task)
                model.save_head('heads_pos/'+task+'_head/', task+"_head")
                best_epoch = epoch
                prev_smallest_dev_loss = cur_epoch_dev_loss

            if epoch%5==0:
                true_labels = torch.flatten(torch.cat(expected_list)).cpu().numpy()
                predicted_labels = torch.flatten(torch.cat(predictions_list)).cpu().numpy()
                print(confusion_matrix(true_labels, predicted_labels))
                print('Micro f1:', f1_score(true_labels, predicted_labels, average='micro'))
                print('Macro f1:', f1_score(true_labels, predicted_labels, average='macro'))
                print('Weighted f1:', f1_score(true_labels, predicted_labels, average='weighted'))

    print('Best epoch:', best_epoch, prev_smallest_dev_loss, task)



# test evaluation

ad = model.load_adapter("adapters_pos/"+task+"_adapter")
model.load_head("heads_pos/"+task+"_head")
model.active_adapters = ad

model.to(device)
model.eval()
predictions_list = []
expected_list = []
for i, batch in enumerate(test_dataloader):
    for k, v in batch.items():
        if k=='pos_input' and len(v)>1:
            batch[k] = torch.stack(v).to(device)
        else:
            batch[k] = v.to(device)
    
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], pos_input=batch["pos_input"], adapter_names=[task])
    predictions = torch.argmax(outputs[0], 1)
    expected = batch["labels"].float()
    predictions_list.append(predictions)
    expected_list.append(expected)
print('Test set evaluation!', task)
true_labels = torch.flatten(torch.cat(expected_list)).cpu().numpy()
predicted_labels = torch.flatten(torch.cat(predictions_list)).cpu().numpy()
print(confusion_matrix(true_labels, predicted_labels))
print('Micro f1:', f1_score(true_labels, predicted_labels, average='micro'))
print('Macro f1:', f1_score(true_labels, predicted_labels, average='macro'))
print('Weighted f1:', f1_score(true_labels, predicted_labels, average='weighted'))


