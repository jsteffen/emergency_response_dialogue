## Usage examples:

**Fine-grained ISO-based annotation (+evaluation):**
```
$ python3 finetune_tradr.py --data_dir=data_iso --mode=speaker --iso_labels=True
$ python3 finetune_tradr.py --evaluation=True --data_dir=data_iso --mode=speaker --output_dir=outputs/8 --iso_labels=True
```

**Coarse-grained Einsatzbefehl-based annotation (+evaluation):**
```
$ python3 finetune_tradr.py --data_dir=data_einsatzbefehl --mode=speaker
$ python3 finetune_tradr.py --evaluation=True --data_dir=data_einsatzbefehl --mode=speaker --output_dir=outputs/8
```

E.g. if you evaluate the model trained with corase-grained Einsatzbefehl labels, your console output may look like this (accuracy per label and the overall statistics with macro F1 for the test data):
```
Kontakt_Anfrage {'total': 35, 'matched': 35} proportion: 0.124 acc: 1.0
Kontakt_Bestaetigung {'total': 33, 'matched': 24} proportion: 0.117 acc: 0.727
Information_geben {'total': 132, 'matched': 124} proportion: 0.468 acc: 0.939
Information_nachfragen {'total': 29, 'matched': 28} proportion: 0.103 acc: 0.966
Absage {'total': 7, 'matched': 2} proportion: 0.025 acc: 0.286
Einsatzbefehl {'total': 21, 'matched': 14} proportion: 0.074 acc: 0.667
Zusage {'total': 23, 'matched': 7} proportion: 0.082 acc: 0.304
Sonstiges {'total': 2, 'matched': 2} proportion: 0.007 acc: 1.0
Total turns: 282
***** Eval results *****
  acc = 0.8368794326241135
  acc_and_f1 = 0.7814700046477632
  eval_loss = 0.5695672863059573
  f1 = 0.726060576671413
```
