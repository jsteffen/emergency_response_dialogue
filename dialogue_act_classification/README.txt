The code in adapters_combined.py performs the dialogue act classification and slot tagging for all orders (Einsatzbefehle) that occur in the text. For each order (Einsatzbefehl) it provides annotations for 5 different slots: Einheit/Auftrag/Mittel/Ziel/Weg.

You can run the code as follows: $ adapters_combined.py

The function 'annotate_turns(intexts, speakers=None, with_previous_turn=True, with_speaker=True, annotate_dact=True, annotate_iso=False, annotate_slots=True)' annotates each turn with the dialogue acts and slots and returns a list with annotations that can be stored as a JSON file (see the function 'write_into_file(turn_annotations, fname)').

As input it expects a list of sentences (no tokenization or lower-casing is needed) and the list of speakers (if available). You can also specify whether you want to include the previous turn and/or speaker for the domain-specific dialogue act annotations and whether you want to annotate fine-graied ISO labels as well.

Usage example:

from adapters_combined import *
turns = ["D5 ist zum Treffpunkt angekommen", "UGV 2 ist bereit", "UGV vorrängig weiter Personen suchen"]
speakers = ["D5", "UGV2", "UGV"]
turn_annotations = annotate_turns(turns, speakers, with_previous_turn=False, with_speaker=False, annotate_iso=True)
fname = "example_annotations.json"
write_into_file(turn_annotations, fname)

The output can be stored in a JSON file that has the following structure:

List of the turn objects where each object has "turn_tokens" (list of Strings), "dialogue_act", "iso_dialogue_act" and "einsatzbefehl_slots" (dictionary).

The dictionary "einsatzbefehl_slots" has the following fields:

"einheit_tags": list of the BIO tags for each token
"einheit_tokens": list of Strings (tokens)

"auftrag_tags": list of the BIO tags for each token
"auftrag_tokens": list of Strings (tokens)

"mittel_tags": list of the BIO tags for each token
"mittel_tokens": list of Strings (tokens)

"ziel_tags": list of the BIO tags for each token
"ziel_tokens": list of Strings (tokens)

"weg_tags": list of the BIO tags for each token
"weg_tokens": list of Strings (tokens)

Sample output can be found in example_annotations.json.

Note that adapters and heads are stored inside heads/ and adapters/ folders.


You can also re-train dialogue act classifier using the following code:
$ python adapters_classifier.py

To prepare the data for the low-resource setting you can run make_low_resource.py

For the data augmentation experiments:

prepare_masks_and_edits.py generates the new data with the substitutions based on the LM predictions and random edits (insert/delete/swap).

prepare_backtranslations.py backtranslates the original text of the training and development sets.

The original data are stored in the csv_da_annotations directory and the augmented data can be either generated with prepare_backtranslations.py and prepare_masks_and_edits.py or provided upon a request (for the sake of reproducibility).
If you need the original augmented data please send an email to: tatiana.anikina@dfki.de
