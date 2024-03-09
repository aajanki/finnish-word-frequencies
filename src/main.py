# Compute word frequencies in c4-fi

import click
import copy
import langid
import json
import os.path
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from urllib.parse import urlparse
from ftfy import fix_text
from spacy.lang.char_classes import ALPHA
from smart_open import open
from .classifiers import SpamClassifier, CodeClassifier
from .finnish import FinnishCustom

@click.command()
@click.option('--destination', default='results',
              help='Directory or S3 path where the results are to be stored')
@click.option('--skip', default=0, help='Number of document to skip')
@click.option('--limit', default=0, help='Maximum number of documents to process')
@click.option('--progress-interval', default=0.1,
              help='Progress bar update interval in seconds')
@click.option('--checkpoint-interval', default=0,
              help='Save checkpoint after this many documents')
def main(destination, skip, limit, progress_interval, checkpoint_interval):
    start_time = datetime.now(tz=timezone.utc)
    spam_classifier = SpamClassifier('models/spam_classifier_weights.json')
    code_classifier = CodeClassifier('models/code_classifier_weights.json')
    tokenize = create_tokenizer()

    if destination:
        print(f'destination = {destination}')
    if skip:
        print(f'skip = {skip}')
    if limit:
        print(f'limit = {limit}')
    if checkpoint_interval:
        print(f'checkpoint interval = {checkpoint_interval}')

    dataset = load_dataset('allenai/c4', 'fi', split='train', streaming=True, trust_remote_code=True)
    if skip > 0:
        dataset = dataset.skip(skip)
    if limit > 0:
        dataset = dataset.take(limit)
    dataset = label_with_index(dataset)
    dataset = tqdm(
        dataset,
        total=limit,
        smoothing=0.02,
        file=sys.stdout,
        mininterval=max(progress_interval, 0.1),
        maxinterval=max(progress_interval, 10)
    )
    dataset = (cleanup_text(x) for x in dataset)
    dataset = (x for x in dataset if is_body_text(x))
    dataset = (x for x in dataset if not is_spam(x, spam_classifier))
    dataset = (x for x in dataset if not is_code(x, code_classifier))
    dataset = (x for x in dataset if is_finnish(x))
    dataset = (cleanup_punctuation(x) for x in dataset)

    num_after_filtering = 0
    num_processed = 0
    wordcounts = Counter()
    for item in dataset:
        num_after_filtering += 1
        num_processed = item['i'] + 1
        text = item['text']
        wordcounts.update(tokenize(text))

        if num_processed % 100000 == 0:
            print(f'Processed {num_processed} documents...')

        if num_after_filtering % 100000 == 0:
            # This will leak memory unless the tokenizer is re-created
            # periodically
            del tokenize
            tokenize = create_tokenizer()

        if checkpoint_interval > 0 and num_processed % checkpoint_interval == 0:
            print(f'Saving a checkpoint after processing {num_processed} documents')
            save_results(
                wordcounts,
                skip,
                limit,
                num_after_filtering,
                num_processed,
                start_time,
                destination,
                f'{num_processed:09d}'
            )

    print(f'Processed {num_processed} documents')
    print(f'{wordcounts.total()} tokens in total, {len(wordcounts)} unique')
    print(f'Writing results to {destination}')

    save_results(wordcounts, skip, limit, num_after_filtering, num_processed, start_time, destination)


def save_results(
        wordcounts,
        skip,
        limit,
        num_after_filtering,
        num_processed,
        start_time,
        destination,
        checkpoint_label=None
):
    if not destination.startswith('s3://'):
        Path(destination).mkdir(parents=True, exist_ok=True)

    if checkpoint_label:
        basename = f'frequencies-mc4-fi-{checkpoint_label}'
    else:
        basename = 'frequencies-mc4-fi'

    with open(os.path.join(destination, f'{basename}.bz2'), 'w') as f:
        for word, freq in wordcounts.most_common():
            f.write(f'{freq}\t{word}\n')

    meta = {
        'timestamp': start_time.isoformat(timespec='seconds'),
        'documents_processed': num_after_filtering,
        'documents_processed_including_filtered': num_processed,
        'total_tokens': wordcounts.total(),
        'unique_tokens': len(wordcounts),
    }
    if skip:
        meta['skip'] = skip
    if limit:
        meta['limit'] = limit
    with open(os.path.join(destination, f'{basename}.meta'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def create_tokenizer():
    tokenizer = FinnishCustom().tokenizer

    def tokenize(text):
        doc = tokenizer(text)
        tokens = (t.text for t in doc)
        tokens = (t for t in tokens if len(t) < 30)
        tokens = (t for t in tokens if not t.isspace())
        return list(tokens)

    return tokenize


def label_with_index(dataset):
    for i, x in enumerate(dataset):
        out = copy.deepcopy(x)
        out['i'] = i
        yield out


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
    text = re.sub(
        r'[\u0000-\u0008\u000E-\u001F\u007F-\u0084\u0086-\u009F\u00AD\u200B-\u200F\u2060\uFEFF\uFFF0-\uFFFF]',
        '',
        text
    )
    text = re.sub(
        r'[\u0530-\u1DBF\u2C00-\uA6FF\uA800-\uAB2F\uAB70-\uD7FF\uE000-\uFAFF\uFB50-\uFDFF]+',
        ' ',
        text
    )
    text = re.sub(r'(?<=\s)[\ufe00-\ufe0f]+', '', text)
    text = re.sub(r'\s\.(?=[{a}]{{4}})'.format(a=ALPHA), '. ', text)
    text = re.sub(r'\.\.\.(?=[-+*/!?%(),:;<>€$£"\'])', '... ', text)
    text = re.sub(r'([][<>"”\'´#*][.,!?])(?=[{a}])'.format(a=ALPHA), r'\1 ', text)
    text = re.sub(r'(?<=[.,!?])([][()<>«»"”\'´#*])', r' \1', text)
    text = re.sub(r'\s+', ' ', text)
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
