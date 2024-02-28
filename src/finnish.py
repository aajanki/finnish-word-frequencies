from spacy.lang.char_classes import ALPHA
from spacy.lang.fi import FinnishDefaults
from spacy.language import Language
from spacy_fi_experimental_web_md.fi import FinnishExDefaults

# Add some tokenizer rules not included in
# spacy_fi_experimental_web_md 0.14

_improved_prefixes = FinnishExDefaults.prefixes + [
    r'\|(?=\w)', '\u200e'
]
_improved_infixes = FinnishExDefaults.infixes + [
    r'\[', r'\]', '<', '>', '«', '»', '·', '•', r'\|', r'(?<=\w)\)(?=\(\w)', r'(?<=\w\))\((?=\w)',
    r'(?<=[{a}]),(?=.)'.format(a=ALPHA), r'(?<=[{a}].),'.format(a=ALPHA),
    r'!(?=[!?])', r'\?(?=!)', r'(?<=!)[!?]', r'(?<=\?)!'
]
_improved_suffixes = FinnishExDefaults.suffixes + [
    r'(?<=\w)\|', r'(?<=\w\w)[\\]', '\u200e', '\u200f'
]

class FinnishCustomDefaults(FinnishDefaults):
    prefixes = _improved_prefixes
    infixes = _improved_infixes
    suffixes = _improved_suffixes
    syntax_iterators = FinnishExDefaults.syntax_iterators

class FinnishCustom(Language):
    lang = 'fi'
    Defaults = FinnishCustomDefaults
