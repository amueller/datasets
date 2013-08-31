import os
from glob import glob

import numpy as np
from matplotlib.colors import ListedColormap
from scipy.misc import imread


data_path = "/home/data/amueller/nyu_depth_forest/fold5/"
mapper = np.zeros(256)
labels = np.array([0, 9, 185, 227, 255])
mapper[labels] = np.array([4, 3, 0, 1, 2])
#mapper[labels] = np.array([0, 2, 3, 1, 4])
#label_colors = [0, 185, 227, 255, 9]


class NYUSegmentation(object):
    def __init__(self, directory=None):
        if directory is None:
            directory = data_path
        self.directory = directory
        self.prediction_path = directory
        self.classes = ["prop", "wall", "furniture", "ground" , "void"]
        path_prefix = os.path.dirname(os.path.realpath(__file__))
        colors = np.loadtxt(os.path.join(path_prefix, "pascal_colors.txt"))
        self.cmap = ListedColormap(colors)
        self.void_label = 5

    def load_image(self, filename):
        return imread(self.directory + "/input/%s_lab_image.png" % filename)

    def get_ground_truth(self, filename):
        image = imread(self.directory + "/prediction_all/%s_lab_image_groundTruth.png"
                       % filename)
        # hack around to get the label integers from the png
        gt_int = mapper[image[:, :, 0].ravel()]
        gt_int = gt_int.reshape(image.shape[:2])
        return gt_int

    def get_split(self, which='train'):
        if which not in ["train", "val"]:
            raise ValueError("Expected 'which' to be 'train' or 'val', got %s."
                             % which)
        if which == "train":
            image_path = "training/"
        else:
            image_path = "validation/"
        files = sorted(glob(self.directory + image_path + "*_lab_image.png"))
        files = [os.path.basename(image_file)[:5] for  image_file in files]
        return files


if __name__ == "__main__":
    pascal = NYUSegmentation()
    from IPython import embed
    embed()
