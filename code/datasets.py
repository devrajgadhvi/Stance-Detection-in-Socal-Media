import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445


def _stance(path, topic=None):
    def clean_ascii(text):
        # function to remove non-ASCII chars from data
        return ''.join(i for i in text if ord(i) < 128)
    orig = pd.read_csv(path, delimiter='\t', header=0, encoding = "latin-1")
    orig['Tweet'] = orig['Tweet'].apply(clean_ascii)
    df = orig
    # Get only those tweets that pertain to a single topic in the training data
    if topic is not None:
        df = df.loc[df['Target'] == topic]
    X = df.Tweet.values
    stances = ["AGAINST", "FAVOR", "NONE", "UNKNOWN"]
    class_nums = {s: i for i, s in enumerate(stances)}
    Y = np.array([class_nums[s] for s in df.Stance])
    return X, Y

def stance(data_dir, topic=None):
    path = Path(data_dir)
    trainfile = 'semeval2016-task6-trainingdata.txt'
    testfile = 'SemEval2016-Task6-subtaskA-testdata.txt'

    X, Y = _stance(path/trainfile, topic=topic)
    teX, _ = _stance(path/testfile, topic=topic)
    tr_text, va_text, tr_sent, va_sent = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print(tr_sent)
    trX = []
    trY = []
    for t, s in zip(tr_text, tr_sent):
        trX.append(t)
        trY.append(s)

    vaX = []
    vaY = []
    for t, s in zip(va_text, va_sent):
        vaX.append(t)
        vaY.append(s)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX, trY), (vaX, vaY), (teX, )

if __name__ == "__main__":
    ## Test
    data_dir = "./data"

    (trX, trY), (vaX, vaY), teX = stance(data_dir)

    print(trX[:5], trY[:5])
    print(len(trX))
    print(len(teX))

