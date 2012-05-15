# This file loads the MNIST - rot dataset as available here:
# http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations
# Should also work with the other variations by adjusting the filename

import numpy as np
from zipfile import ZipFile


def load_mnist_rot(path=None, which='train'):
    if path == None:
        path = "/home/local/datasets/bengio/mnist_rotation_new.zip"
    f = ZipFile(path)
    train_file_name = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file_name = 'mnist_all_rotation_normalized_float_test.amat'
    print(f.namelist())
    if which == 'train':
        file_name = train_file_name
    elif which == 'test':
        file_name = test_file_name
    else:
        raise ValueError("'which' must be either 'train' or 'test'")

    with f.open(file_name) as data_file:
        data = np.loadtxt(data_file)
    X = data[:, :-1]
    y = data[:, -1].astype(np.int)
    return X, y


def test_mnist_rot():
    import matplotlib.pyplot as plt
    X, y = load_mnist_rot()
    plt.figure()
    for i in xrange(1, 11):
        plt.subplot(2, 5, i)
        plt.imshow(X[i, :].reshape(28, 28))
        plt.title(y[i])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_mnist_rot()
