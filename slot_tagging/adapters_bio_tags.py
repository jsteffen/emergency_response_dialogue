import datasets
from datasets import load_dataset
from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
import copy
import sys
import os

os.environ["WANDB_DISABLED"] = "true"
label_type = "neg_samples" #"all_samples" "neg_samples"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_len_bio = 128
data_type = "balanced" # "all_samples"
adapters_dir = "adapters_" + data_type
heads_dir = "heads_" + data_type
do_train = False

def encode_labels(example):
    r_tags = []
    count = 0
    all_tokenized = []
    for idx, token in enumerate(example["tokens"].split()):
        tokenized = tokenizer.tokenize(token)
        all_tokenized.extend(tokenized)
        label = example["tags"].split()[idx]
        for ti, t in enumerate(tokenized):
            if ti!=0:
                if label=="O":
                    r_tags.append(label2id[label])
                else:
                    r_tags.append(label2id["I"])
            else:
                r_tags.append(label2id[label])
    r_tags = [label2id["O"]]+r_tags[:max_len_bio-2]+[label2id["O"]]# for CLS and SEP tokens
    rest = max_len_bio-len(r_tags)
    if rest>0:
        for i in range(rest):
            r_tags.append(label2id["O"])
    labels = dict()
    labels["labels"] = torch.tensor(r_tags)
    return labels

def encode_data(data):
    encoded = tokenizer([doc for doc in data["tokens"]], pad_to_max_length=True, padding="max_length", max_length=max_len_bio, truncation=True, add_special_tokens=True)
    return (encoded)

for task in ["einheit", "auftrag", "mittel", "ziel", "weg"]:
    labels = ["B", "I", "O"]
    id2label = {id_: label for id_, label in enumerate(labels)}
    label2id = {label: id_ for id_, label in enumerate(labels)}

    model_name = "bert-base-german-cased"
    config = AutoConfig.from_pretrained(model_name, num_label=len(labels), id2label=id2label, label2id=label2id, layers=2)
    model = AutoAdapterModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if do_train:
        model.add_adapter(task)
        model.add_tagging_head(task+"_head", num_labels=len(labels), id2label=id2label)

        train_dataset = datasets.Dataset.from_csv(label_type+"_csv/"+label_type+"_"+task+"_train.csv")
        train_dataset = train_dataset.map(encode_labels)
        train_task_dataset = train_dataset.map(encode_data, batched=True, batch_size=16)

        dev_dataset = datasets.Dataset.from_csv(label_type+"_csv/"+label_type+"_"+task+"_dev.csv")
        dev_dataset = dev_dataset.map(encode_labels)
        dev_task_dataset = dev_dataset.map(encode_data, batched=True, batch_size=16)

        test_task_dataset = datasets.Dataset.from_csv(label_type+"_csv/"+label_type+"_"+task+"_test.csv")
        test_task_dataset = test_task_dataset.map(encode_labels)
        test_task_dataset = test_task_dataset.map(encode_data, batched=True, batch_size=16)

        train_task_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        dev_task_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        test_task_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

        dataloader = torch.utils.data.DataLoader(train_task_dataset, shuffle=True)
        evaluate_dataloader = torch.utils.data.DataLoader(dev_task_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_task_dataset)

        model.to(device)
        model.set_active_adapters([[task]])
        model.train_adapter([task])
    
        class_weights = torch.FloatTensor([1.5, 1.5, 1.0]).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-4,}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=1e-3)

        prev_smallest_dev_loss = None
        best_epoch = None

        for epoch in range(12):
            for i, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], adapter_names=[task])
                predictions = torch.flatten(outputs[0], 0, 1)
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
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch["input_ids"], adapter_names=[task])
                    predictions = torch.argmax(outputs[0], 2)
                    expected = batch["labels"].float()

                    mpredictions = torch.flatten(outputs[0], 0, 1)
                    mexpected = torch.flatten(batch["labels"].long(), 0, 1)
                    loss = loss_function(mpredictions, mexpected)
                    dev_losses.append(loss.item())
            
                    predictions_list.append(predictions)
                    expected_list.append(expected)
                cur_epoch_dev_loss = round(sum(dev_losses)/len(dev_losses),3)
                print(epoch, "Dev loss:", cur_epoch_dev_loss)
                if prev_smallest_dev_loss is None or cur_epoch_dev_loss<=prev_smallest_dev_loss:
                    # save adapter and head
                    model.save_adapter(adapters_dir+"/"+task+"_adapter/", task)
                    model.save_head(heads_dir+"/"+task+"_head/", task+"_head")
                    best_epoch = epoch
                    prev_smallest_dev_loss = cur_epoch_dev_loss

                if epoch%5==0:
                    true_labels = torch.flatten(torch.cat(expected_list)).cpu().numpy()
                    predicted_labels = torch.flatten(torch.cat(predictions_list)).cpu().numpy()
                    print(confusion_matrix(true_labels, predicted_labels))
                    print("Micro f1:", f1_score(true_labels, predicted_labels, average="micro"))
                    print("Macro f1:", f1_score(true_labels, predicted_labels, average="macro"))
                    print("Weighted f1:", f1_score(true_labels, predicted_labels, average="weighted"))
                

        print("Best epoch:", best_epoch, prev_smallest_dev_loss, task)

        from transformers import TokenClassificationPipeline
        print("task:", task)
        tagger = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task="auftrag", device=0)
        intext = "UGV sollte zur Rauchentwicklung ins Obergeschoss über die Wiese fahren"
        res = tagger(intext)
        print(res)
    else:
        # test evaluation
        print("Task:", task)
        test_task_dataset = datasets.Dataset.from_csv(label_type+"_csv/"+label_type+"_"+task+"_test.csv")
        test_task_dataset = test_task_dataset.map(encode_labels)
        test_task_dataset = test_task_dataset.map(encode_data, batched=True, batch_size=16)

        test_task_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        test_dataloader = torch.utils.data.DataLoader(test_task_dataset)

        ad = model.load_adapter(adapters_dir+"/"+task+"_adapter")
        model.load_head(heads_dir+"/"+task+"_head")
        model.active_adapters = ad

        model.to(device)
        model.eval()
        predictions_list = []
        expected_list = []
        for i, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], adapter_names=["adapters/"+task+"_adapter"])
            predictions = torch.argmax(outputs[0], 2)
            expected = batch["labels"].float()        
            predictions_list.append(predictions)
            expected_list.append(expected)
        print("Test set evaluation!")
        true_labels = torch.flatten(torch.cat(expected_list)).cpu().numpy()
        predicted_labels = torch.flatten(torch.cat(predictions_list)).cpu().numpy()
        print(confusion_matrix(true_labels,predicted_labels))
        print("Micro f1:", f1_score(true_labels, predicted_labels, average="micro"))
        print("Macro f1:", f1_score(true_labels, predicted_labels, average="macro"))
        print("Weighted f1:", f1_score(true_labels, predicted_labels, average="weighted"))
    


