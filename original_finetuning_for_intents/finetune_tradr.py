#!/usr/bin/env python

# Usage example:
# Fine-grained ISO-based annotations:
# $ python3 finetune_tradr.py --data_dir=data_iso --mode=speaker --iso_labels=True
# $ python3 finetune_tradr.py --evaluation=True --data_dir=data_iso --mode=speaker --output_dir=outputs/8 --iso_labels=True

# Coarse-grained Einsatzbefehl-based annotations:
# $ python3 finetune_tradr.py --data_dir=data_einsatzbefehl --mode=speaker
# $ python3 finetune_tradr.py --evaluation=True --data_dir=data_einsatzbefehl --mode=speaker --output_dir=outputs/8

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, tqdm_notebook, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, iso_labels):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(np.unicode(cell, "utf-8") for cell in line)
                lines.append(line)

            return lines


class TradrProcessor(DataProcessor):
    """Processor for the TRADRZ data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, iso_labels):
        if iso_labels:
            return [
                "Promise",
                "AcceptRequest",
                "Answer",
                "Confirm",
                "Agreement",
                "Retraction",
                "SetQuestion",
                "Disagreement",
                "Communicative Function",
                "Auto-negative",
                "TurnAssign",
                "Inform",
                "AcceptOffer",
                "InteractionStructuring",
                "FeedbackElicitation",
                "TurnAccept",
                "SelfError",
                "Suggestion",
                "CheckQuestion",
                "Offer",
                "ChoiceQuestion",
                "AcceptSuggestion",
                "Interaction Structuring",
                "DeclineOffer",
                "AddressOffer",
                "TurnTake",
                "SelfCorrection",
                "DeclineRequest",
                "Question",
                "Instruct",
                "Pausing",
                "PropositionalQuestion",
                "Disconfirm",
                "TurnRelease",
                "Allo-positive",
                "Thanking",
                "Auto-positive",
                "Stalling",
                "Opening",
                "AddressRequest",
                "Feedback Elicitation",
                "Request",
            ]
        else:
            return [
                "Kontakt_Anfrage",
                "Kontakt_Bestaetigung",
                "Einsatzbefehl",
                "Information_geben",
                "Information_nachfragen",
                "Zusage",
                "Absage",
                "Sonstiges",
            ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        prev_turn = ""
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_b = line[1]
            if args.mode == "only":
                text_a = None  # single-sentence training
            elif args.mode == "speaker":
                text_a = line[0]  # speaker + sentence
            elif args.mode == "prev":
                text_a = prev_turn  # prev sentence + sentence
            else:
                raise Exception(
                    f"Unknown mode {args.mode}! Can be one of the following: only, speaker, prev"
                )
            label = line[2]
            prev_turn = text_b
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_b = tokenizer.tokenize(example.text_b)
        tokens_a = None
        if example.text_a:
            tokens_a = tokenizer.tokenize(example.text_a)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            _truncate_seq_pair([], tokens_b, max_seq_length - 2)

        sent = 0
        tokens = ["[CLS]"]
        segment_ids = [sent]
        if tokens_a:
            tokens += tokens_a + ["[SEP]"]
            segment_ids = [sent] * len(tokens)
            sent += 1

        tokens += tokens_b + ["[SEP]"]
        segment_ids += [sent] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 0:
            print("*** Example ***")
            print("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = TradrProcessor()

    label_list = processor.get_labels(args.iso_labels)
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    # BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = (
        int(1 + len(train_examples) / args.train_batch_size) * args.num_train_epochs
    )

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model, num_labels=num_labels
    )
    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nod in n for nod in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nod in n for nod in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_train_optimization_steps,
    )

    global_step = 0

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer
    )

    print(f"Num examples = {len(train_examples)}")
    print(f"Batch size = {args.train_batch_size}")
    print(f"Num steps = {num_train_optimization_steps}")

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    evdata = get_eval_data(
        processor,
        tokenizer,
        processor.get_dev_examples(args.data_dir),
        args.max_seq_length,
        args.iso_labels,
    )
    loss_fct = CrossEntropyLoss()

    model.train()
    epoch = 0
    for _ in range(args.num_train_epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask)
            logits = logits[0]

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        out_dir = args.output_dir + "/" + str(epoch)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(
            "Epoch {} Training loss total {} per example {} ".format(
                epoch, tr_loss, tr_loss / nb_tr_examples
            )
        )
        model.train()
        eval_model(
            device,
            tokenizer,
            model,
            args.mode,
            evdata,
            out_dir,
            args.iso_labels,
            args.max_seq_length,
        )

        print(f"Saving model checkpoint to {out_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        epoch += 1
    return model


def get_eval_data(processor, tokenizer, eval_examples, max_seq_length, iso_labels):
    label_list = processor.get_labels(iso_labels)
    num_labels = len(label_list)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer
    )

    print("  Num examples = {}".format(len(eval_examples)))

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return eval_data, num_labels, all_label_ids


def load_and_eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    processor = TradrProcessor()

    label_list = processor.get_labels(args.iso_labels)
    num_labels = len(label_list)

    ev_data = get_eval_data(
        processor,
        tokenizer,
        processor.get_test_examples(args.data_dir),
        args.max_seq_length,
        args.iso_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir, num_labels=num_labels
    )
    model = model.to(device)
    eval_model(
        device,
        tokenizer,
        model,
        args.mode,
        ev_data,
        args.output_dir,
        args.iso_labels,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        what="test",
    )


def eval_model(
    device,
    tokenizer,
    model,
    mode,
    evdat,
    output_dir,
    iso_labels,
    max_seq_length=128,
    eval_batch_size=8,
    what="eval",
    verbose=False,
):
    eval_data = evdat[0]
    num_labels = evdat[1]
    all_label_ids = evdat[2]

    label_map = {label: i for i, label in enumerate(TradrProcessor().get_labels(iso_labels))}
    reversed_label_map = {v: k for k, v in label_map.items()}

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    print("  Batch size = {}".format(eval_batch_size))

    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    prev_labels = ["None"] * eval_batch_size
    label_distribution = dict()
    total_turns = 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits[0]

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        predicted = np.argmax(logits.detach().cpu().numpy(), axis=1)
        prev_labels = [reversed_label_map[p] for p in predicted]

        for i in range(len(input_ids)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i][: sum(input_mask[i])])
            assigned = reversed_label_map[predicted[i]]
            true_label = reversed_label_map[label_ids[i].item()]
            total_turns += 1
            if true_label != assigned and verbose:
                print(tokens)
                print("Assigned:", assigned)
                print("True:", true_label)
                print()

            if assigned not in label_distribution:
                label_distribution[assigned] = {"total": 0, "matched": 0}
            if true_label not in label_distribution:
                label_distribution[true_label] = {"total": 0, "matched": 0}
            label_distribution[true_label]["total"] += 1
            if true_label == assigned:
                label_distribution[true_label]["matched"] += 1

    print("Label distribution and accuracy per label")
    all_labels = sum([label_distribution[lbl]["total"] for lbl in label_distribution])
    for lbl in label_distribution:
        acc_score = label_distribution[lbl]["matched"] / max(1, label_distribution[lbl]["total"])
        lbl_proportion = label_distribution[lbl]["total"] / max(1, all_labels)
        print(
            lbl,
            label_distribution[lbl],
            "proportion:",
            round(lbl_proportion, 3),
            "acc:",
            round(acc_score, 3),
        )
    print("Total turns:", total_turns)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    result = acc_and_f1(preds, all_label_ids.numpy())

    result["eval_loss"] = eval_loss

    output_eval_file = os.path.join(output_dir, what + "_results.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print(f"  {key} = {result[key]}")
            writer.write(f"{key} = {result[key]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--iso_labels", type=bool, default=False)
    parser.add_argument("--evaluation", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--bert_model", type=str, default="dbmdz/bert-base-german-cased")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=0)

    args = parser.parse_args()
    if args.evaluation:
        # evaluate saved model
        print("Evaluating...")
        load_and_eval_model(args)
    else:
        # train a new model
        print("Training...")
        print(args.iso_labels)
        model = train_model(args)
