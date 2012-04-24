import numpy as np
import os
import cPickle

from joblib import Memory
memory = Memory(cachedir="cache", verbose=5)


def unpickle(file):
    with open(file, 'rb') as fo:
        mydict = cPickle.load(fo)
    fo.close()
    return mydict


def load_batch(batch, path):
    if batch != "test":
        batch_file = os.path.join(path, "data_batch_" + str(batch))
    else:
        batch_file = os.path.join(path, "test_batch")
    batch_pickle = unpickle(batch_file)
    return batch_pickle['data'], batch_pickle['labels']


@memory.cache
def cifar_data(split="train", path="cifar-10-batches-py"):
    """ returns data, labels """
    if split == "train":
        batches = np.arange(1, 6)
        data = []
        labels = []
        for batch in batches:
            batch_data, batch_labels = load_batch(batch)
            data.append(batch_data)
            labels.append(batch_labels)
        return np.vstack(data), np.hstack(labels)
    elif split == "test":
        batch_data, batch_labels = load_batch("test")
        return np.array(batch_data), np.array(batch_labels)
    else:
        raise ValueError("'split' should be one of 'train', 'test'")
