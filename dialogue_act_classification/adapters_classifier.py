import datasets
from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig, BertConfig, BertModelWithHeads, AdapterConfig
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
import sys
import os

from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

task = "dact"
anno_type = "without_context_with_current_speaker"
# valid annotation types:
# without_context_and_without_speaker
# without_context_with_current_speaker
# with_context_with_current_and_previous_speaker
# iso
# iso_simplified
# summary
# low_resource_turn_and_speaker

do_training = False
batch_size = 32
model_name = "bert-base-german-cased"
data_folder = "csv_da_annotations/csv_"+anno_type
low_resource_annotation_prefix = ""
# valid low-resource annotation types:
# "" corresponds to the baseline (no data augmentation)
# backtranslated_
# backtranslated_with_fr_
# (masked|random)_(0.1|0.2|0.4|0.6)_(1|2|5|10)_
# e.g.: "masked_0.2_5_"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAdapterModel.from_pretrained(model_name)

label2id = {"Absage":0, "Einsatzbefehl":1, "Information_geben":2, "Information_nachfragen":3, "Kontakt_Anfrage":4, "Kontakt_Bestaetigung":5, "Sonstiges":6, "Zusage":7}
id2label = dict()
for k,v in label2id.items():
    id2label[v] = k

def encode_data(data):
    if anno_type=="without_context_and_without_speaker":
        encoded = tokenizer([doc for doc in data["tokens"]], pad_to_max_length=True, padding="max_length", max_length=128, truncation=True, add_special_tokens=True)
    elif anno_type=="without_context_with_current_speaker" or anno_type=="low_resource_turn_and_speaker":
        tokenizer([data["speakers"][doc_i]+" [SEP] "+doc_tokens for doc_i, doc_tokens in enumerate(data["tokens"])], pad_to_max_length=True, padding="max_length", max_length=256, truncation=True, add_special_tokens=True)
    elif anno_type=="with_context_with_current_and_previous_speaker":
        tokenizer([data["previous_speakers"][doc_i]+" [SEP] "+data["previous"][doc_i]+" [SEP] "+data["speakers"][doc_i]+" [SEP] "+doc_tokens for doc_i, doc_tokens in enumerate(data["tokens"])], pad_to_max_length=True, padding="max_length", max_length=256, truncation=True, add_special_tokens=True)
    elif anno_type=="iso_simplified" or anno_type=="iso":
        encoded = tokenizer([data["speakers"][doc_i]+" [SEP] "+data["isoda"][doc_i]+" [SEP] "+doc_tokens for doc_i, doc_tokens in enumerate(data["tokens"])], pad_to_max_length=True, padding="max_length", max_length=256, truncation=True, add_special_tokens=True)
    elif anno_type=="summary":
        encoded = tokenizer([truncate_summary(data["summary"][doc_i])+" [SEP] "+data["speakers"][doc_i]+" [SEP] "+doc_tokens for doc_i, doc_tokens in enumerate(data["tokens"])], pad_to_max_length=True, padding="max_length", max_length=512, truncation=True, add_special_tokens=True)

    return (encoded)


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def compute_f1(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(preds)):
        predicted = preds[i]
        gold = p.label_ids[i]
        if gold==predicted:
            tp+=1
        elif gold!=predicted and gold==1:
            fn+=1
        elif gold!=predicted and gold==0:
            fp+=1
    if tp+fp>0:
        prec = tp/(tp+fp)
    else:
        prec = 0
    if tp+fn>0:
        rec = tp/(tp+fn)
    else:
        rec = 0
    if prec+rec>0:
        f1 = 2*prec*rec/(prec+rec)
    else:
        f1 = 0
    return {"f1": f1}

# training the model

if do_training:
    config = AdapterConfig.load("pfeiffer")
    model.add_adapter(task, config=config)

    model.add_classification_head(task+"_head", num_labels=len(label2id), id2label=id2label, use_pooler=True)
    model.train_adapter(task)

    train_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+low_resource_annotation_prefix+"train.csv")
    train_task_dataset = train_task_dataset.map(encode_data, batched=True, batch_size=batch_size)
    train_task_dataset = train_task_dataset.rename_column("tokens","text").rename_column("tags","labels")

    dev_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+low_resource_annotation_prefix+"dev.csv")
    dev_task_dataset = dev_task_dataset.map(encode_data, batched=True, batch_size=batch_size)
    dev_task_dataset = dev_task_dataset.rename_column("tokens","text").rename_column("tags","labels")

    train_task_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_task_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model.to(device)


    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=100,
        output_dir="training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )


    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_task_dataset,
        eval_dataset=dev_task_dataset,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    print(trainer.evaluate())
    
    if anno_type=="low_resource_turn_and_speaker":
        model.save_adapter("adapters/low_resource_adapters/"+task+"_"+low_resource_annotation_prefix.replace(".","-")+"adapter/", task)
        model.save_head("heads/low_resource_heads/"+task+"_"+low_resource_annotation_prefix.replace(".","-")+"head/", task+"_head")
    else:
        model.save_adapter("adapters/"+task+"_adapter/", task)
        model.save_head("heads/"+task+"_head/", task+"_head")

    model = None

# test evaluation

from transformers import TextClassificationPipeline
intexts = []
gold_labels = []
test_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+"test.csv")
for i in range(len(test_task_dataset)):
    if anno_type=="iso_simplified" or anno_type=="iso":
        intexts.append(test_task_dataset["speakers"][i]+" [SEP] "+test_task_dataset["isoda"][i]+" [SEP] "+test_task_dataset["tokens"][i])
    elif anno_type=="summary":
        if test_task_dataset["summary"][i] is not None:
            truncated_summary = " ".join(test_task_dataset["summary"][i].split()[-250:])
        else:
            truncated_summary = "Start"
        intexts.append(truncated_summary+" [SEP] "+test_task_dataset["speakers"][i]+" [SEP] "+test_task_dataset["tokens"][i])
    if anno_type=="without_context_and_without_speaker":
        intexts.append(test_task_dataset["tokens"][i])
    elif anno_type=="without_context_with_current_speaker" or anno_type=="low_resource_turn_and_speaker":
        intexts.append(test_task_dataset["speakers"][i]+" [SEP] "+test_task_dataset["tokens"][i])
    elif anno_type=="with_context_with_current_and_previous_speaker":
        intexts.append(test_task_dataset["previous_speakers"][i]+" [SEP] "+test_task_dataset["previous"][i]+" [SEP] "+test_task_dataset["speakers"][i]+" [SEP] "+test_task_dataset["tokens"][i])
    gold_labels.append(test_task_dataset["tags"][i])

if anno_type=="low_resource_turn_and_speaker":
    adapter = model.load_adapter("adapters/low_resource_adapters/"+task+"_"+low_resource_annotation_prefix.replace(".","-")+"adapter/")
    head = model.load_head("heads/low_resource_heads/"+task+"_"+low_resource_annotation_prefix.replace(".","-")+"head/")
else:
    adapter = model.load_adapter("adapters/"+task+"_"+anno_type+"/"+task+"_adapter")
    head = model.load_head("heads/"+task+"_"+anno_type+"/"+task+"_head")

model.active_adapters = adapter
dact_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, task=task, device=0 if device=="cuda" else -1)

all_labels = ["Absage", "Einsatzbefehl", "Information_geben", "Information_nachfragen", "Kontakt_Anfrage", "Kontakt_Bestaetigung", "Sonstiges", "Zusage"]

scores = dict()
for label in all_labels:
    scores[label] = {"tp":0, "fp":0, "fn":0}
match = 0
for i, intext in enumerate(intexts):
    predicted_label = ""
    prediction = dact_classifier(intext)
    if len(prediction)>0:
        predicted_label = prediction[0]["label"]
    gold_label = id2label[gold_labels[i]]
    
    if predicted_label==gold_label:
        match+=1
        scores[predicted_label]["tp"]+=1
    else:
        #print(prediction, intext, ">>>", predicted_label, gold_label)
        scores[predicted_label]["fp"]+=1
        scores[gold_label]["fn"]+=1
        
print("Accuracy:", round(match/len(intexts),3), "matched:", match, "total:", len(intexts))
print("F1 scores:")
f1scores = 0
# compute f1 scores (per label)
for label in all_labels:
    tp = scores[label]["tp"]
    fp = scores[label]["fp"]
    fn = scores[label]["fn"]
    if tp+fp>0:
        prec = tp/(tp+fp)
    else:
        prec = 0
    if tp+fn>0:
        rec = tp/(tp+fn)
    else:
        rec = 0
    if prec+rec>0:
        f1score = 2*prec*rec/(prec+rec)
    else:
        f1score = 0
    f1scores+=f1score
    print(label, "F1:", round(f1score,3))
# compute macro f1 score (avg)
print("Macro F1:", round(f1scores/len(all_labels),3))    

