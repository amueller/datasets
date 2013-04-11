import os
from glob import glob

import numpy as np
import Image


classes = ['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
           'aeroplane', 'water', 'face', 'car', 'bicycle', 'flower',
           'sign', 'bird', 'book', 'chair', 'road', 'cat', 'dog',
           'body', 'boat', 'void', 'horse', 'mountain']


colors = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
          [0, 128, 128], [128, 128, 128], [192, 0, 0], [64, 128, 0],
          [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
          [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
          [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
          [192, 64, 0], [0, 0, 0], [128, 0, 128], [64, 0, 0]]


class MSRCDataset(object):
    def __init__(self,  directory=None, rm_mountain_horse=True):
        if directory is None:
            directory = "/home/local/datasets/MSRC_ObjCategImageDatabase_v2/"
        self.directory = directory
        images = glob(os.path.join(self.directory, "Images", "*.bmp"))
        self.images = [os.path.basename(f)[:-4] for f in images]
        self.n_images = len(self.images)

        if self.n_images == 0:
            raise ValueError("no images found in directory %s", self.directory)

        self.convert = [99. / 1000,  587. / 1000,  114. / 1000]
        converted_colors = np.dot(np.array(colors), self.convert).tolist()
        label_dict = dict()
        if rm_mountain_horse:
            horse_idx = classes.index("horse")
            mountain_idx = classes.index("mountain")
            void_idx = classes.index("void")
            horse_color = converted_colors[horse_idx]
            mountain_color = converted_colors[mountain_idx]
            label_dict[horse_color] = void_idx
            label_dict[mountain_color] = void_idx
            converted_colors.remove(horse_color)
            converted_colors.remove(mountain_color)
            colors.pop(mountain_idx)
            colors.pop(horse_idx)
            classes.remove("horse")
            classes.remove("mountain")
        for i, color in enumerate(converted_colors):
            label_dict[color] = i
        self.label_dict = label_dict
        self.n_classes = len(classes)
        self.classes = np.array(classes)
        self.colors = np.array(colors)

    def get_images(self):
        return [self.get_image(image) for image in self.images]

    def get_image(self, image):
        f = os.path.join(self.directory, "Images", "%s.bmp" % image)
        return np.array(Image.open(f))

    def get_ground_truth(self, image):
        f = os.path.join(self.directory, "GroundTruth", "%s_GT.bmp" % image)
        img = Image.open(f)
        img = np.array(img)
        dp = np.dot(img, self.convert)
        labels = np.unique(dp)  # find all occuring labels
        img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        for l in labels:
            val = self.label_dict[l]
            img[dp == l] = val
        return img

    def get_split(self, which="train"):
        path_prefix = os.path.dirname(os.path.realpath(__file__))
        if which == "train":
            files = np.loadtxt(os.path.join(path_prefix, "msrc_train.txt"),
                               dtype=np.str)
        elif which == "val":
            files = np.loadtxt("msrc_validation.txt")
        elif which == "test":
            files = np.loadtxt("msrc_test.txt")
        else:
            raise ValueError("Expected 'which' to be one of 'train', 'val',"
                             "'test', got %s." % which)
        return [f[:-4] for f in files]


if __name__ == "__main__":
    msrc = MSRCDataset()
    from IPython import embed
    embed()
