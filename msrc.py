import os
from glob import glob

from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import numpy as np
import Image


_classes = ['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
            'aeroplane', 'water', 'face', 'car', 'bicycle', 'flower',
            'sign', 'bird', 'book', 'chair', 'road', 'cat', 'dog',
            'body', 'boat', 'void', 'horse', 'mountain']
classes = np.array(_classes)

_colors = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
          [0, 128, 128], [128, 128, 128], [192, 0, 0], [64, 128, 0],
          [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
          [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
          [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
          [192, 64, 0], [0, 0, 0], [128, 0, 128], [64, 0, 0]]
colors = np.array(_colors)


class MSRC21Dataset(object):
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
            horse_idx = np.where(classes == "horse")[0]
            mountain_idx = np.where(classes == "mountain")[0]
            void_idx = np.where(classes == "void")[0]
            horse_color = converted_colors[horse_idx]
            mountain_color = converted_colors[mountain_idx]
            label_dict[horse_color] = void_idx
            label_dict[mountain_color] = void_idx
            converted_colors.remove(horse_color)
            converted_colors.remove(mountain_color)
            colors_ = colors.tolist()
            colors_.pop(mountain_idx)
            colors_.pop(horse_idx)
            classes_ = classes.tolist()
            classes_.remove("horse")
            classes_.remove("mountain")
        for i, color in enumerate(converted_colors):
            label_dict[color] = i
        self.label_dict = label_dict
        self.n_classes = len(classes_)
        self.classes = np.array(classes_)
        self.colors = np.array(colors_)
        self.void_label = 21
        self.cmap = ListedColormap(self.colors)

    def get_images(self):
        return [self.get_image(image) for image in self.images]

    def get_image(self, image):
        f = os.path.join(self.directory, "Images", "%s.bmp" % image)
        return np.array(Image.open(f))

    def get_ground_truth(self, image, ds="old"):
        if ds == "old":
            f = os.path.join(self.directory, "GroundTruth", "%s_GT.bmp" %
                             image)
            img = Image.open(f)
            img = np.array(img)
            dp = np.dot(img, self.convert)
            labels = np.unique(dp)  # find all occuring labels
            img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            for l in labels:
                val = self.label_dict[l]
                img[dp == l] = val
        elif ds == "new":
            if self.n_classes == 22:
                # removed horse and mountain
                labels = np.arange(1, 24).tolist()
                labels.pop(7)  # mountain
                labels.pop(4)  # horse
                labels.append(0)
                labels = np.array(labels)
                label_map = 21 * np.ones(24, dtype=np.int)
                label_map[labels] = np.arange(22)
            else:
                # don't doing this right now...
                # never gonna need it
                raise NotImplementedError
            f = os.path.join(self.directory, "newsegmentations_mats",
                             "%s_GT.mat" % image)
            mat_file = loadmat(f)
            img = mat_file['newseg']
            gt_labels = mat_file['newlabels'].ravel()
            n_segments = len(np.unique(img))
            if n_segments > len(gt_labels):
                # add void label
                gt_labels = np.hstack([gt_labels, [0]])
            assert(n_segments == len(gt_labels))
            img = label_map[gt_labels[img - 1]]
        else:
            raise ValueError("Expected ds='old' or 'new', got %s." % ds)
        return img

    def get_split(self, which="train"):
        path_prefix = os.path.dirname(os.path.realpath(__file__))
        if which == "train":
            thefile = "msrc_train.txt"
        elif which == "val":
            thefile = "msrc_validation.txt"
        elif which == "test":
            thefile = "msrc_test.txt"
        else:
            raise ValueError("Expected 'which' to be one of 'train', 'val',"
                             "'test', got %s." % which)
        files = np.loadtxt(os.path.join(path_prefix, thefile),
                           dtype=np.str)
        return [f[:-4] for f in files]

    def eval_pixel_performance(self, file_names, prediction_images,
                               print_results=True):
        # doesn't work if horse and mountain are still in.
        from sklearn.metrics import confusion_matrix
        confusion = np.zeros((22, 22))
        for prediction, f in zip(prediction_images, file_names):
            # load ground truth image
            gt = self.get_ground_truth(f)
            confusion += confusion_matrix(gt.ravel(), prediction.ravel(),
                                          labels=np.arange(0, 22))
        # drop void
        confusion_normalized = (confusion.astype(np.float) /
                                confusion.sum(axis=1)[:, np.newaxis])
        per_class_acc = np.diag(confusion_normalized)[:-1]
        global_acc = np.diag(confusion)[:-1].sum() / confusion[:-1, :].sum()
        average_acc = np.mean(per_class_acc)
        if print_results:
            print("global: %.4f, average: %.4f" % (global_acc, average_acc))
            print(["%s: %.2f" % (c, x)
                   for c, x in zip(classes, per_class_acc)])
        return {'global': global_acc, 'average': average_acc,
                'per_class': per_class_acc, 'confusion': confusion}


if __name__ == "__main__":
    msrc = MSRC21Dataset()
    from IPython import embed
    embed()
