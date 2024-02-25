# Compute word frequencies in mc4-fi

import copy
import langid
import re
import unicodedata
import spacy_fi_experimental_web_md
from collections import Counter
from itertools import islice
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from urllib.parse import urlparse
from ftfy import fix_text
from spacy.lang.char_classes import ALPHA
from .classifiers import SpamClassifier, CodeClassifier

def main():
    max_texts = 1000
    result_path = Path('results')
    spam_classifier = SpamClassifier('models/spam_classifier_weights.json')
    code_classifier = CodeClassifier('models/code_classifier_weights.json')
    tokenize = create_tokenizer()

    result_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('mc4', 'fi', split='train', streaming=True, trust_remote_code=True)
    dataset = (cleanup_text(x) for x in dataset)
    dataset = (x for x in dataset if is_body_text(x))
    dataset = (x for x in dataset if not is_spam(x, spam_classifier))
    dataset = (x for x in dataset if not is_code(x, code_classifier))
    dataset = (x for x in dataset if is_finnish(x))
    dataset = (cleanup_punctuation(x) for x in dataset)

    doc_count = 0
    wordcounts = Counter()
    for item in tqdm(islice(dataset, max_texts), total=max_texts, smoothing=0.02):
        doc_count += 1
        text = item['text']
        wordcounts.update(tokenize(text))

    print(f'Processed {doc_count} documents')
    print(f'{wordcounts.total()} tokens in total, {len(wordcounts)} unique')

    with open(result_path / 'frequencies-mc4-fi', 'w') as f:
        for word, freq in wordcounts.most_common():
            f.write(f'{freq}\t{word}\n')


def create_tokenizer():
    tokenizer = spacy_fi_experimental_web_md.fi.FinnishExtended().tokenizer

    def tokenize(text):
        doc = tokenizer(text)
        tokens = (t.text for t in doc)
        # Ignore tokens that are mostly digits: dates, times, prices, etc.
        #tokens = (t for t in tokens if sum(x.isdigit() for x in t) < len(t)/2)
        tokens = (t for t in tokens if len(t) < 30)
        return list(tokens)

    return tokenize


def cleanup_text(x):
    text = x['text']
    text = unicodedata.normalize('NFC', text.strip())
    text = fix_text(text, uncurl_quotes=False)
    text = remove_sort_entry(text)
    out = copy.deepcopy(x)
    out['text'] = text
    return out


def cleanup_punctuation(x):
    text = re.sub(r'[\s\u2800]+', ' ', x['text'])
    text = re.sub(r'[\u0000-\u0008\u000E-\u001F\u007F-\u0084\u0086-\u009F\u00AD\u200B-\u200D\u2060\uFEFF\uFFF0-\uFFFF]', '', text)
    text = re.sub(r'[\u0530-\u1DBF\u2C00-\uA6FF\uA800-\uAB2F\uAB70-\uD7FF\uE000-\uFAFF\uFB50-\uFDFF]+', ' ', text)
    text = re.sub(r'(?<=\s)[\ufe00-\ufe0f]+', '', text)
    text = re.sub(r'\s\.(?=[{a}]{{4}})'.format(a=ALPHA), '. ', text)
    text = re.sub(r'\.\.\.(?=[-+*/!?%(),:;<>€$£"\'])', '... ', text)
    out = copy.deepcopy(x)
    out['text'] = text
    return out


def is_spam(x, spam_classifier):
    return is_spam_url(x['url']) or spam_classifier.predict(x['text'])


def is_code(x, code_classifier):
    return code_classifier.predict(x['text'])


def is_body_text(x):
    tokens = x['text'].split()
    num_tokens = len(tokens)
    if num_tokens == 0:
        return False

    num_digit_tokens = sum(1 for t in tokens if re.match(r'^\(?[0-9][0-9.,:;]+\)?$', t))
    if num_digit_tokens / num_tokens > 0.25:
        return False

    num_single_character_tokens = sum(1 for t in tokens if len(t) == 1)
    if num_single_character_tokens / num_tokens > 0.25:
        return False

    return True


def is_finnish(x):
    langcode, _ = langid.classify(x['text'])
    return langcode == 'fi'


def remove_sort_entry(text):
    """Remove SortEntry code block appearing in many e-commerce sites."""
    return re.sub(r'loc_fi_FI, sid_[A-Z0-9]+, prod, sort_\[SortEntry\(order=[A-Z_]+, direction=[A-Z_]+\)]', ' ', text)

skip_tlds = set([
    '.nl',
    '.be',
    '.one',
    '.pw',
])

def is_spam_url(url):
    dom = domain(url)
    tld = '.' + dom.split('.')[-1]
    return tld in skip_tlds


def domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    # Remove port
    netloc = netloc.split(':', 1)[0]
    return netloc


if __name__ == '__main__':
    main()
