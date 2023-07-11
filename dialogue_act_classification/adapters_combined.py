from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig
from transformers import TextClassificationPipeline, TokenClassificationPipeline
import torch
import json

device = 0 if torch.cuda.is_available() else -1

max_len_bio = 128

model_name = "bert-base-german-cased"
model = AutoAdapterModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# remove extra spaces and commas
def normalize_text(texts):
    normalized_texts = []
    for text in texts:
        text = ' '.join(text.split())
        text = text.replace(',','') # remove commas because they were removed from the training data
        normalized_texts.append(text)
    return normalized_texts

# make the BIO tags continous, each tag must start with B
def remap_annotations(annotation, original_text):
    position2token = dict()
    token_start = 0
    splitted_tokens = original_text.split()
    for token_i, token in enumerate(splitted_tokens):
        for i in range(token_start, token_start+len(token)):
            position2token[i] = token_i
        token_start+=len(token)+1 # add one to account for the space
    token2annotation = dict()
    for anno in annotation:
        token_i = position2token[anno['start']]
        if not(token_i in token2annotation):
            token2annotation[token_i] = anno['entity']
    annotated_tokens = []
    slot_tokens = []
    for token_i in range(len(splitted_tokens)):
        if token_i in token2annotation:
            annotated_tokens.append(token2annotation[token_i])
            slot_tokens.append(splitted_tokens[token_i])
        else:
             annotated_tokens.append('O')
    return annotated_tokens, slot_tokens

# add speaker and/or previous turn to the input turns
def add_speaker_prev_turn(intexts, speakers, add_prev_turn):
    new_intexts = []
    assert(len(intexts)==len(speakers))
    prev_turn = 'Start'
    for i, turn in enumerate(intexts):
        if add_prev_turn:
            new_text = prev_turn+' [SEP] '+speakers[i]+' [SEP] '+turn
            prev_turn = turn
        else:
            new_text = speakers[i]+' [SEP] '+turn
        new_intexts.append(new_text)
        
    return new_intexts

# annotate turns with dialogue acts, ISO labels and slots for the orders (Einsatzbefehl)
# input: intexts: list of turns (each turn is a String)
# Dialogue act labels:
# Kontakt_Anfrage, Kontakt_Bestaetigung, Information_nachfragen, Information_geben, Einsatzbefehl, Zusage, Absage, Sonstiges
# Slot labels:
# Einheit, Auftrag, Mittel, Ziel, Weg
# ISO labels:
# Answer, Disconfirm, Inform, Request, Offer, Confirm, Auto-positive, TurnAccept, Question, TurnAssign, PropositionalQuestion, Agreement, Promise, TurnTake, AddressRequest, AcceptOffer, AcceptRequest, Pausing, CheckQuestion, DeclineOffer, Auto-negative, SetQuestion, ChoiceQuestion, Instruct, Allo-positive, Other
def annotate_turns(intexts, speakers=None, with_previous_turn=True, with_speaker=True, annotate_dact=True, annotate_iso=False, annotate_slots=True):
    turn_annotations = []
    intexts = normalize_text(intexts)    

    # domain-specific dialogue act classification
    if annotate_dact:
        # prepare model input, adapter and head if only the speaker is used (no previous turn)
        if with_speaker and not(with_previous_turn):
            adapter = model.load_adapter("adapters/dact_without_context_with_current_speaker/dact_adapter")
            model.load_head("heads/dact_without_context_with_current_speaker/dact_head")
            if speakers is None:
                raise Exception("For this setting speakers must be specified!")
            else:
               dact_intexts = add_speaker_prev_turn(intexts, speakers, add_prev_turn=False)
        # prepare model input, adapter and head if both the speaker and the previous turn are used
        elif with_speaker and with_previous_turn:
            adapter = model.load_adapter("adapters/dact_adapter_with_context_with_current_and_previous_speaker/dact_adapter")
            model.load_head("heads/dact_adapter_with_context_with_current_and_previous_speaker/dact_head")
            if speakers is None:
                raise Exception("For this setting speakers must be specified!")
            else:
                dact_intexts = add_speaker_prev_turn(intexts, speakers, add_prev_turn=True)
        # prepare model input, adapter and head if only the turn text is available
        elif not(with_previous_turn) and not(with_speaker):
            adapter = model.load_adapter("adapters/dact_without_context_and_without_speaker/dact_adapter")
            model.load_head("heads/dact_without_context_and_without_speaker/dact_head")
            dact_intexts = intexts
        else:
            raise Exception("Incorrect setting! You can choose between PreviousTurn+Speaker+CurrentTurn, Speaker+CurrentTurn and CurrentTurn")
        model.active_adapters = adapter
        dact_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, task="dact", device=device)
        for i_text, dact_intext in enumerate(dact_intexts):
            turn_annotation = dict()
            turn_annotation['turn_tokens'] = intexts[i_text].split()
            turn_annotation['dialogue_act'] = dact_classifier(dact_intext)[0]['label']
            turn_annotation['einsatzbefehl_slots'] = dict()
            turn_annotations.append(turn_annotation)

    # ISO dialogue act classification
    if annotate_iso:
        adapter = model.load_adapter('adapters/dact_iso/dact_adapter')
        model.load_head('heads/dact_iso/dact_head')
        model.active_adapters = adapter
        iso_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, task="iso", device=device)
        for i_text, intext in enumerate(intexts):     
            turn_annotations[i_text]['iso_dialogue_act'] = iso_classifier(intext)[0]['label']

    # slot tagging for Einsatzbefehl
    if annotate_slots:
        tasks = ['auftrag', 'einheit', 'mittel', 'ziel', 'weg']
        for task in tasks:
            adapter = model.load_adapter("../slot_tagging/adapters_balanced/"+task+"_adapter")
            model.load_head("../slot_tagging/heads_balanced/"+task+"_head")
            model.active_adapters = adapter
            slot_tagger = TokenClassificationPipeline(model=model, tokenizer=tokenizer,  task=task, device=device)
            for i_text, intext in enumerate(intexts):
                annotation = slot_tagger(intext)
                annotated_tokens, slot_tokens = remap_annotations(annotation, intext)
                turn_annotations[i_text]['einsatzbefehl_slots'][task+'_tags'] = annotated_tokens
                turn_annotations[i_text]['einsatzbefehl_slots'][task+'_tokens'] = slot_tokens
    
    return turn_annotations

# write annotations into json file
def write_into_file(turn_annotations, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(turn_annotations, indent = 2))

# code execution example
def run_example():
    intexts = ["D5 ist zum Treffpunkt angekommen", "UGV 2 ist bereit", "UGV vorrängig weiter Personen suchen"]
    speakers = ["D5", "UGV2", "UGV"]
    turn_annotations = annotate_turns(intexts, speakers, with_previous_turn=False, with_speaker=False, annotate_iso=True)
    for turn_annotation in turn_annotations:
        for k,v in turn_annotation.items():
            print(k, v)
    fname = "example_annotations.json"
    write_into_file(turn_annotations, fname)

if __name__ == "__main__":
    run_example()
