import os
from glob import glob

import numpy as np
from matplotlib.colors import ListedColormap
from scipy.misc import imread


data_path = "/home/data/amueller/nyu_depth_forest/fold1/"
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
        self.classes = ["structure", "prop", "furniture", "ground" , "void"]
        #path_prefix = os.path.dirname(os.path.realpath(__file__))
        #colors = np.loadtxt(os.path.join(path_prefix, "pascal_colors.txt"))
        colors = np.array([[185, 202, 202], [227, 172, 0], [255, 94, 28],
                           [9, 68, 83], [0, 0, 0]]) / 255.
        self.cmap = ListedColormap(colors)
        self.void_label = 4

    def get_image(self, filename):
        return imread(self.directory + "/input/%s_lab_image.png" % filename)

    def get_depth(self, filename):
        depth_file = self.directory + "/input/%s_depth_image.data" % filename
        f = open(depth_file)
        shape = np.fromfile(f, dtype=np.uint16, count=2)
        depth_image = np.fromfile(f, dtype=np.float32).reshape(shape[::-1])
        return depth_image

    def get_pointcloud_normals(self, file_name):
        f = ("%s/input/%s_depth_image.data.pcl"
             % (self.directory, file_name))
        if os.path.exists(f + "_.npy"):
            return np.load(f + "_.npy")
        else:
            point_cloud = np.loadtxt(f, skiprows=11)
            normals = point_cloud[:, :6]
            normals =  normals.reshape(480, 640, 6)
            np.save(f + "_.npy", normals)
        return normals

    def get_ground_truth(self, filename):
        image = imread(self.directory + "/prediction_all/%s_lab_image_groundTruth.png"
                       % filename)
        # hack around to get the label integers from the png
        gt_int = mapper[image[:, :, 0].ravel()]
        gt_int = gt_int.reshape(image.shape[:2])
        return gt_int.astype(np.int)

    def get_split(self, which='train'):
        if which not in ["train", "val", 'trainval', "test"]:
            raise ValueError("Expected 'which' to be 'train', 'val', 'trainval' or 'test', got %s."
                             % which)
        if which == "train":
            image_path = "training/"
        elif which == 'val':
            image_path = "validation/"
        elif which == 'test':
            image_path = "../images_test/"
        elif which == 'trainval':
            return np.unique(self.get_split(which='train') + self.get_split(which='val'))

        files = sorted(glob(self.directory + image_path + "*_lab_image.png"))
        files = [os.path.basename(image_file)[:5] for  image_file in files]
        return files


if __name__ == "__main__":
    pascal = NYUSegmentation()
    from IPython import embed
    embed()
