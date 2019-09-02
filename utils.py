from pathlib import Path

import unicodedata


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def strip_accents(s):
    """Replaces accentual characters with ASCII counterparts, or removes them if counterpart not present"""
    return str(''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').encode('ascii', 'ignore'), 'utf-8')
