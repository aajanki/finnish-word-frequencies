# Finnish word frequencies

Counting the word frequencies in the Finnish subset of the MC4 dataset.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.main
```

## Text classifiers

The models directory contains simple models for detecting spam and
computer code. They are used to filter out uninteresting documents.

The models have been trained using the scripts at src/trainmodels with
manually labelled training samples.

TODO: add training data
