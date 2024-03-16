# pseudonymize email addresses in a vocabulary file

import hashlib
import re
import sys
from smart_open import open
from pathlib import Path

sensitive_pattern = re.compile(
    r'\.[A-Za-z0-9_%+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9-]|.@gmail\.[A-Za-z0-9_%+.-]+|.@yahoo\.[A-Za-z0-9_%+.-]+|.@hotmail\.[A-Za-z0-9_%+.-]+',
    re.IGNORECASE
)
etunimi_sukunimi_pattern = re.compile(
    r'^(?:etunimi\.sukunimi|etu\.sukunimi|sukunimi\.etunimi|firstname\.lastname)@',
    re.IGNORECASE
)

def main():
    destination = Path('results')
    frequencies = []

    with open(sys.argv[1]) as f:
        for line in f:
            [freq, token] = line.rstrip('\n').split('\t', 1)

            if sensitive_pattern.search(token) and not etunimi_sukunimi_pattern.search(token):
                [username, domain] = token.split('@', 1)
                h = hashlib.md5()
                h.update(username.encode('utf-8'))
                digest = h.hexdigest()[:16]
                token = f'redacted-{digest}@{domain}'

            frequencies.append([int(freq), token])

    frequencies = sorted(frequencies, key=lambda x: (-x[0], x[1]))
    with open(destination / f'pseudonymized.bz2', 'w') as f:
        for freq, token in frequencies:
            f.write(f'{freq}\t{token}\n')


if __name__ == '__main__':
    main()
