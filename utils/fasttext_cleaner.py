# -*- coding: utf-8 -*-

"""
Cleaning the text in the same way as FastText pre-trained models
Inspired by https://github.com/SignalN/ftio/blob/master/ftio/wiki/preproc.py
modified to use regex (faster) instead of sed subprocess.
"""

import re


def __special_chars(s):
    s = re.sub('’', "'", s)
    s = re.sub('′', "'", s)
    s = re.sub("''", " ", s)
    s = re.sub("'", " ' ", s)
    s = re.sub('“', '"', s)
    s = re.sub('”', '"', s)

    s = re.sub('"', ' ', s)
    s = re.sub('\\.', ' . ', s)
    s = re.sub('<br \\/>', ' ', s)
    s = re.sub(', ', ' , ', s)

    s = re.sub('\\(', ' ( ', s)
    s = re.sub('\\)', ' ) ', s)
    s = re.sub('\\!', ' ! ', s)
    s = re.sub('\\?', ' ? ', s)

    s = re.sub('\\;', ' ', s)
    s = re.sub('\\:', ' ', s)
    s = re.sub('-', ' - ', s)
    s = re.sub('=', ' ', s)
    s = re.sub('=', ' ', s)

    s = re.sub('\\*', ' ', s)
    s = re.sub('\\|', ' ', s)
    s = re.sub('\\«', ' ', s)

    return s


def __whitespace(s):
    return ' '.join(s.split())


def __digits(s):
    s = ''.join(filter(lambda c: not c.isdigit(), s))
    return re.sub('\s+', ' ', s).strip()


def fasttext_cleaning(s):
    return __digits(__whitespace(__special_chars(s.lower())))
