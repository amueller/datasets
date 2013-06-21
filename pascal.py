import numpy as np
from matplotlib.colors import ListedColormap
from scipy.misc import imread


pascal_path = "/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011"


class PascalSegmentation(object):
    def __init__(self, directory=None):
        if directory is None:
            directory = pascal_path
        self.directory = directory
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'potted plant', 'sheep', 'sofa', 'train', 'tvmonitor',
                        'void']
        colors = np.loadtxt("pascal_colors.txt")
        self.cmap = ListedColormap(colors)
        self.void_label = 255

    def get_image(self, filename):
        return imread(self.directory + "/JPEGImages/%s.jpg" % filename)

    def get_ground_truth(self, filename):
        return imread(self.directory + "/SegmentationClass/%s.png" % filename)

    def labels(self):
        return [self.class_num()]

    def get_split(self, which='train', year="2010"):
        if which not in ["train", "val", "train1", "train2"]:
            raise ValueError("Expected 'which' to be 'train' or 'val', got %s."
                             % which)
        split_file = self.directory + "/ImageSets/Segmentation/%s.txt" % which
        files = np.loadtxt(split_file, dtype=np.str)
        return [f for f in files if f.split("_")[0] <= year]


if __name__ == "__main__":
    pascal = PascalSegmentation()
    from IPython import embed
    embed()
