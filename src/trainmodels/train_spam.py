# Train a simple spam classifier

import json
import unicodedata
import re
import numpy as np
from itertools import repeat
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from joblib import dump

def main():
    documents = json.load(open('trainingdata/opetus.json'))
    test_documents = json.load(open('trainingdata/spam_opetus_002.json'))

    data = load_doc_features(documents, Path('trainingdata/documents/3'), min_df=3)
    test_data = load_doc_features(test_documents, Path('trainingdata/documents/2'))

    print(f'Loaded {sum(data["spam"])} spam documents, '
          f'{len(data["documentId"]) - sum(data["spam"])} non-spam messages')

    pipe = train_model(data)

    print('# Evaluation #')
    evaluate_model(pipe, data, 'training')
    evaluate_model(pipe, test_data, 'test')

    dump(pipe, 'spam_classifier.joblib')

    pruned_model_size = 100
    important_features = extract_important_features(
        pipe.named_steps['classifier'],
        pipe.named_steps['vectorizer'].get_feature_names_out(),
        pruned_model_size
    )
    print('important features')
    print(important_features)

    feature_vec = pipe.named_steps['vectorizer'].get_feature_names_out().tolist()
    selected_features = important_features['spam'] + important_features['ham']
    selected_features_idx = [feature_vec.index(i) for i in selected_features]

    clf = pipe.named_steps['classifier']
    for i in range(clf.n_features_in_):
        if i not in selected_features_idx:
            clf.feature_log_prob_[:, i] = 0

    print('# Evaluation: pruned model #')
    evaluate_model(pipe, data, 'training')
    print()
    evaluate_model(pipe, test_data, 'test')

    dump(pipe, 'spam_classifier_pruned.joblib')

    pruned = prune_model(
        pipe.named_steps['classifier'],
        pipe.named_steps['vectorizer'].get_feature_names_out(),
        pruned_model_size
    )

    with open('spam_classifier_weights.json', 'w') as fp:
        json.dump(pruned, fp, ensure_ascii=False)


def train_model(data):
    pipe = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('classifier', MultinomialNB(alpha=1.5, force_alpha=True))
    ])
    pipe.fit(data['features'], data['spam'])

    print(f'{len(pipe.named_steps["vectorizer"].feature_names_)} features')

    return pipe

def evaluate_model(pipe, data, train_or_test):
    idx_spam = (np.array(data['spam']) == 1).nonzero()[0].tolist()
    features_spam = [data['features'][i] for i in idx_spam]
    idx_ham = (np.array(data['spam']) != 1).nonzero()[0].tolist()
    features_ham = [data['features'][i] for i in idx_ham]

    print(f'{train_or_test} accuracy (n = {len(idx_spam)}), spam: {pipe.score(features_spam, np.ones(len(idx_spam)))}')
    print(f'{train_or_test} accuracy (n = {len(idx_ham)}), ham: {pipe.score(features_ham, np.zeros(len(idx_ham)))}')
    print(f'{train_or_test} accuracy, combined (n = {len(data["spam"])}): {pipe.score(data["features"], data["spam"])}')
    print('Confusion matrix:')
    print(confusion_matrix(pipe.predict(data["features"]), data["spam"]))
    

def load_doc_features(documents, document_path, allowed_features=None, min_df=1):
    docids = []
    classes = []
    features = []
    df = {}
    docs_spam = list(zip(documents['spam'], repeat(1)))
    docs_ham = list(zip(documents['ham'], repeat(0)))
    docs = docs_spam + docs_ham
    for docid, class_index in docs:
        docids.append(docid)
        classes.append(class_index)

        with open(document_path / docid, 'r') as f:
            doc_json = json.load(f)
            text = unicodedata.normalize('NFC', doc_json['text'].strip())

            words = (x.group(0) for x in re.finditer(r'\w[-\w]*\w', text.lower()))
            features_in_doc = {
                word: 1
                for word in words
                if not len(word) >= 30
            }

            if allowed_features is not None:
                features_in_doc = {
                    feat: 1 for feat in features_in_doc.keys()
                    if feat in allowed_features
                }

            features.append(features_in_doc)

            for f in features_in_doc:
                df[f] = df.get(f, 0) + 1

    if min_df > 1:
        selected_features = set()
        for f, count in df.items():
            if count >= min_df:
                selected_features.add(f)

        cleaned_features = []
        for features_in_doc in features:
            cleaned_features.append({
                f: x for (f, x) in features_in_doc.items() if f in selected_features
            })
    else:
        cleaned_features = features

    return {
        'documentId': docids,
        'spam': classes,
        'features': cleaned_features
    }

def extract_important_features(classifier, feature_names, top_n=10):
    logp = classifier.feature_log_prob_

    top_indices = np.abs(logp[0, :] - logp[1, :]).argsort()[-top_n:][::-1]

    top_indices_class_0 = []
    top_indices_class_1 = []
    for i in top_indices:
        if logp[0, i] > logp[1, i]:
            top_indices_class_0.append(i)
        else:
            top_indices_class_1.append(i)

    top_words_class_0 = [feature_names[i] for i in top_indices_class_0]
    top_words_class_1 = [feature_names[i] for i in top_indices_class_1]

    return {'ham': top_words_class_0, 'spam': top_words_class_1}


def prune_model(classifier, feature_names, top_n=10):
    logp = classifier.feature_log_prob_
    top_indices = np.abs(logp[0, :] - logp[1, :]).argsort()[-top_n:][::-1]
    weights = {feature_names[i]: logp[0, i] - logp[1, i] for i in top_indices}
    bias = classifier.class_log_prior_[0] - classifier.class_log_prior_[1]
    
    return {
        "weights": weights,
        "bias": bias,
    }


if __name__ == '__main__':
    main()
