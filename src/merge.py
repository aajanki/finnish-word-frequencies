# Merge frequencies and metadata from multiple checkpoints

import json
import sys
from smart_open import open
from pathlib import Path
from collections import Counter

def main():
    destination = Path('results')
    merged_frequencies = {}
    merged_meta = {}

    for p in sys.argv[1:]:
        path = Path(p)
        print(f'Reading {path}')

        with open(path / 'frequencies-mc4-fi.bz2') as f:
            for line in f:
                [c, token] = line.strip('\n').split('\t', 1)
                merged_frequencies[token] = merged_frequencies.get(token, 0) + int(c)

        with open(path / 'frequencies-mc4-fi.meta') as f:
            meta = json.load(f)

        merged_meta['total_tokens'] = merged_meta.get('total_tokens', 0) + meta['total_tokens']

        # Early versions did not generate the "including_filtered" count
        if 'documents_processed_including_filtered' in meta:
            merged_meta['documents_processed'] = \
                merged_meta.get('documents_processed', 0) + meta['documents_processed']
            merged_meta['documents_processed_including_filtered'] = \
                merged_meta.get('documents_processed_including_filtered', 0) + \
                meta['documents_processed_including_filtered']
        else:
            merged_meta['documents_processed'] = \
                merged_meta.get('documents_processed', 0) + meta['limit']
            merged_meta['documents_processed_including_filtered'] = \
                merged_meta.get('documents_processed_including_filtered', 0) + \
                meta['documents_processed']

    merged_meta['unique_tokens'] = len(merged_frequencies)

    with open(destination / f'merged.bz2', 'w') as f:
        freq = sorted(merged_frequencies.items(), key=lambda x: (-x[1], x[0]))
        for token, freq in freq:
            f.write(f'{freq}\t{token}\n')

    with open(destination / f'merged.meta', 'w') as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
