# Train a model that detects if input text is Javascript, JSON, HTML
# or some other computer language instead of Finnish.

import json
import unicodedata
import re
import numpy as np
from itertools import repeat, permutations
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif, mutual_info_classif
from joblib import dump


def main():
    documents = json.load(open('trainingdata/js_eka.json'))
    data = load_texts(documents, Path('trainingdata/documents/1'))
    data = featurize(data)

    test_documents = json.load(open('trainingdata/js_opetus_002.json'))
    test_data = load_texts(test_documents, Path('trainingdata/documents/2'))
    test_data = featurize(test_data)
    
    print(f'Loaded {sum(x == 1 for x in data["classes"])} code documents, '
          f'{sum(x == 0 for x in data["classes"])} non-code documents')

    pipe = train_model(data)

    print('# Evaluation #')
    evaluate_model(pipe, data, 'training')
    evaluate_model(pipe, test_data, 'test')

    if 'feature_selection' in pipe.named_steps:
        feature_names = pipe.named_steps['feature_selection'].get_feature_names_out(
            pipe.named_steps['vectorizer'].feature_names_)
    else:
        feature_names = pipe.named_steps['vectorizer'].feature_names_
        
    weights = model_weights(
        pipe.named_steps['classifier'],
        feature_names
    )

    with open('js_classifier_weights.json', 'w') as fp:
        json.dump(weights, fp, ensure_ascii=False)

    dump(pipe, 'js_classifier.joblib')


def train_model(data):
    #recall0_score = make_scorer(recall_score, pos_label=0)
    pipe = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('feature_selection', SequentialFeatureSelector(
            MultinomialNB(alpha=1.0),
            n_features_to_select=160,
            scoring='f1',
            cv=5,
            n_jobs=4
        )),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    pipe.fit(data['features'], data['classes'])

    if 'feature_selection' in pipe.named_steps:
        print(f'Number of features: {sum(pipe.named_steps["feature_selection"].get_support())} '
              f'(original: {pipe.named_steps["feature_selection"].n_features_in_})')
    else:
        print(f'Number of features: {len(pipe.named_steps["vectorizer"].feature_names_)}')

    return pipe

def evaluate_model(pipe, data, train_or_test):
    idx_koodi = (np.array(data['classes']) == 1).nonzero()[0].tolist()
    features_koodi = [data['features'][i] for i in idx_koodi]
    idx_teksti = (np.array(data['classes']) != 1).nonzero()[0].tolist()
    features_teksti = [data['features'][i] for i in idx_teksti]

    print(f'{train_or_test} accuracy (n = {len(idx_koodi)}), koodi: {pipe.score(features_koodi, np.ones(len(idx_koodi)))}')
    print(f'{train_or_test} accuracy (n = {len(idx_teksti)}), teksti: {pipe.score(features_teksti, np.zeros(len(idx_teksti)))}')
    print(f'{train_or_test} accuracy, combined (n = {len(data["classes"])}): {pipe.score(data["features"], data["classes"])}')
    print('Confusion matrix:')
    print(confusion_matrix(pipe.predict(data["features"]), data["classes"]))

def load_texts(documents, document_path):
    docids = []
    classes = []
    texts = []
    docs = list(zip(documents['koodi'], repeat(1))) + list(zip(documents['teksti'], repeat(0)))
    for docid, class_index in docs:
        docids.append(docid)
        classes.append(class_index)

        with open(document_path / docid, 'r') as f:
            doc_json = json.load(f)
            text = unicodedata.normalize('NFC', doc_json['text'].strip())
            texts.append(text)

    return {
        'documentId': docids,
        'classes': classes,
        'texts': texts
    }

def featurize(data):
    data['features'] = []
    for text in data['texts']:
        data['features'].append(extract_features(text))
    
    return data

unigrams = '( ) [ ] { } < > = - + " \' ; : ! & \\ | _'.split()
bigrams = list(x[0]+x[1] for x in permutations(unigrams, 2))
trigrams = list(x[0]+x[1]+x[2] for x in permutations(unigrams, 3))
feature_names = unigrams + bigrams + trigrams

def extract_features(text):
    features = {}
    
    # word count
    plain_words = re.finditer(r'\b\w\w+\b', text)
    word_count = max(sum(1 for _ in plain_words), 1)
    features['word_count'] = word_count

    # punctuation features
    text_no_space = re.sub(r'\s+', '', text)
    for feat in feature_names:
        c = text_no_space.count(feat)
        crel = round(c / len(text) * 1000)
        if c > 0:
            features[feat] = c
        if crel > 0:
            features['relative ' + feat] = crel

    # html tags
    html_feature_regexs = {
        '<script>': r'<script[ >]|</script>',
        '<iframe>': r'<iframe[ >]|</frame>',
        '<ul>': r'</?ul>',
        '<li>': r'</?li>',
    }
    for feat, html_tag_re in html_feature_regexs.items():
        tags = re.finditer(html_tag_re, text, re.IGNORECASE)
        c = sum(1 for _ in tags)
        crel = round(c / len(text) * 1000)
        if c > 0:
            features[feat] = c
        if crel > 0:
            features['relative ' + feat] = crel

    return features


def model_weights(classifier, feature_names):
    logp = classifier.feature_log_prob_
    weights = {feature_names[i]: logp[0, i] - logp[1, i] for i in range(classifier.n_features_in_)}
    bias = classifier.class_log_prior_[0] - classifier.class_log_prior_[1]

    return {
        "weights": weights,
        "bias": bias,
    }

if __name__ == '__main__':
    main()
