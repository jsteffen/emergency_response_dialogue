#!/usr/bin/env python3


"""
Web service with endpoint for annotation of radio traffic with task specific entities
"""

import argparse
import json
import logging
import string

import torch
from adapters import BertAdapterModel, AutoAdapterModel
from flask import Flask, abort, Response, request, Request
from flask.typing import ResponseReturnValue
from transformers import BertTokenizerFast, AutoTokenizer, AutoConfig
from waitress import serve

# configure logger
logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=True)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Flask app
app = Flask(__name__)

# model related stuff
model_name: str = "bert-base-german-cased"
model: BertAdapterModel
tokenizer: BertTokenizerFast
max_len_bio = 128
labels = ["B", "I", "O"]
id2label = {id_: label for id_, label in enumerate(labels)}
label2id = {label: id_ for id_, label in enumerate(labels)}
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# supported tasks
tasks: list[str] = ["einheit", "auftrag", "mittel", "ziel", "weg"]

# folders
data_type: str = "balanced"  # "all_samples"
adapters_dir: str = "adapters_" + data_type
heads_dir: str = "heads_" + data_type


@app.route('/alive')
def alive() -> ResponseReturnValue:
    """
    Simple check if server is alive
    :return: alive message
    """
    return Response("tag server is alive", status=200, mimetype='text/html')


@app.route('/annotate', methods=['GET', 'POST'])
def annotate() -> ResponseReturnValue:
    """
    Top level entry point to annotate radio traffic with task specific entities
    :return: task specific entities in JSON format
    """
    try:
        text = _get_text_from_request(request)
    except Exception as e:
        logger.error(e)
        abort(400, description=e)
    lines = text.splitlines()
    result = []
    for line in lines:
        line_result = _annotate_line(line)
        if line_result:
            result.append(line_result)
    return Response(json.dumps(result), status=200, mimetype='application/json')


def _get_text_from_request(req: Request) -> str:
    """
    Extract text from request.
    :return: text, linebreaks replaced by space
    """
    if req.method == 'GET':
        if 'text' not in req.args:
            raise ValueError("missing text")
        text = req.args.get('text', type=str, default='')
    elif req.method == 'POST':
        if req.mimetype != 'multipart/form-data':
            raise ValueError(f"invalid content-type '{req.mimetype}', must be multipart/form-data")
        encoding = req.args.get('encoding', type=str, default='utf-8').lower()
        file = req.files['file']
        text = file.read().decode(encoding)
    else:
        raise ValueError(f"unsupported method '{req.method}'")
    return text


def tokenize(line: str):
    all_tokens = ['#BOS']
    for token in line.split():
        tokenized = tokenizer.tokenize(token)
        all_tokens.extend(tokenized)
    return all_tokens


def merge_labels(pred_labels, subtokens):
    current_labels = set()
    merged_labels = []
    for idx, tok in enumerate(subtokens):
        # skip first subtoken, as it's a beginning-of-sentence marker
        if idx == 0:
            continue
        if tok.startswith('##'):
            # add to current token label set
            current_labels.add(pred_labels[idx])
        else:
            # new token start found
            # first determine the label of the current token,
            if 'B' in current_labels:
                # B overrides I
                merged_labels.append('B')
            elif 'I' in current_labels:
                # label dependes on the previous label:
                # if O -> B, if B or I -> I
                if merged_labels[-1] == 'O':
                    merged_labels.append('B')
                else:
                    merged_labels.append('I')
            else:
                merged_labels.append('O')
            current_labels.clear()
            current_labels.add(pred_labels[idx])
    # handle last token
    if 'B' in current_labels:
        merged_labels.append('B')
    elif 'I' in current_labels:
        if merged_labels[-1] == 'O':
            merged_labels.append('B')
        else:
            merged_labels.append('I')
    else:
        merged_labels.append('O')
    return merged_labels[1:]


def _annotate_line(line: str) -> dict[str, dict[str, list[str]]]:
    clean_line = line.translate(str.maketrans('', '', string.punctuation))
    subtokens = tokenize(clean_line)
    encoded_line = tokenizer(clean_line, pad_to_max_length=True, padding="max_length",
                             max_length=max_len_bio,
                             truncation=True, add_special_tokens=True)
    tensor = torch.tensor([encoded_line['input_ids']])
    logger.info(f'processing "{clean_line}"..')
    result = {}
    for task in tasks:
        # set adapter and head for current task
        model.active_adapters = task + "_adapter"
        model.active_head = task + "_head"

        # get predications
        model_result = model(tensor, attention_mask=torch.tensor([encoded_line['attention_mask']]))
        predictions = torch.argmax(model_result[0], 2)[0].tolist()

        # merge subtokens and labels
        merged_labels = merge_labels([id2label[k] for k in predictions], subtokens)

        # group tagged tokens into tagged phrases
        phrases = []
        current_phrase = ''
        for la, to in zip(merged_labels, clean_line.split()):
            if la == 'B':  # beginning of a new phrase
                if current_phrase:  # if a phrase is already being built, add it to phrases
                    phrases.append(current_phrase)
                current_phrase = to  # start a new phrase with the current token
            elif la == 'I':  # continuation of the current phrase
                current_phrase += ' ' + to
            else:  # if it's an 'O' (outside), finalize the current phrase (if any)
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = ''
        # add the last phrase if it exists
        if current_phrase:
            phrases.append(current_phrase)
        if phrases:
            logger.info(f'{task} phrases: {phrases}')
            result[task] = phrases

    return {'text': line, 'phrases': result}


def start_server(port: int, host: str) -> None:
    """
    The main function
    :param port: server port, None if not provided
    :param host: server host ip, None if not provided
    """

    # init model
    logger.info(f"initializing model...")
    AutoConfig.from_pretrained(model_name, num_label=len(labels), id2label=id2label,
                               label2id=label2id, layers=2)
    global model
    model = AutoAdapterModel.from_pretrained(model_name)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for task in tasks:
        # load adapters and heads
        model.load_adapter(adapters_dir + "/" + task + "_adapter", load_as=task + "_adapter")
        model.load_head(heads_dir + "/" + task + "_head", load_as=task + "_head")
    model.to(device)
    model.eval()
    logger.info("model initialized")

    # start server
    if not port:
        port = 5050
    if not host:
        host = '0.0.0.0'
    serve(app, host=host, port=port)


def parse_arguments() -> argparse.Namespace:
    """
    Read command line arguments
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho', '--host', help="server host ip (optional, default 0.0.0.0")
    parser.add_argument('-p', '--port', help="server port (optional, default 5050)")
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    # read command-line arguments and pass them to main function
    args = parse_arguments()
    start_server(args.host, args.port)
