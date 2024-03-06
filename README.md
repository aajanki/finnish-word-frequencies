# Finnish word frequencies

A script for counting the word frequencies in the Finnish subset of
[the C4 dataset](https://huggingface.co/datasets/allenai/c4).

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.main --limit 10000
```

In Docker:

```
docker build --network host --tag fi-vocabulary:latest .
docker volume create fi-frequencies
docker volume inspect fi-frequencies
docker run -it --rm --mount source=fi-frequencies,target=/app/results --dns 8.8.8.8 fifrequencies:latest python -m src.main --limit 10000
```

## Text classifiers

The models directory contains simple models for detecting spam and
computer code. They are used to filter out uninteresting documents.

The models have been trained using the scripts at src/trainmodels with
manually labelled training samples.

TODO: add training data
