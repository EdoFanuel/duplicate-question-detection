import numpy as np
import pandas as pd
import datetime
import operator
import sklearn
import nltk
import matplotlib.pyplot as plt
import pylab


def char_len(text):
    return len(text)


def word_tokenize(text):
    return nltk.word_tokenize(text)
