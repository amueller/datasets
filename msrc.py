import os
from glob import glob

import numpy as np
import Image

#from image_data import ImageDataset


def generate_val_train_msrc(directory, random_init):
    import random
    with open(os.path.join(directory, 'images.txt')) as f:
        image_names = f.readlines()
    class_images = [[] for i in xrange(1, 9)]
    for image_name in image_names:
        class_images[int(image_name[0]) - 1].append(image_name)
    with open(os.path.join(directory, 'train_%d.txt' % random_init), 'w') as train_list:
        with open(os.path.join(directory, 'val_%d.txt' % random_init), 'w') as val_list:
            for class_list in class_images:
                random.shuffle(class_list)
                train_list.writelines(class_list[:len(class_list) / 2])
                val_list.writelines(class_list[len(class_list) / 2:])


class MSRCDataset(object):
    def __init__(self,  directory):
        self.directory = directory
        self.classes = np.array(['void', 'building', 'grass', 'tree', 'cow', 'horse', 'sheep',
                'sky', 'mountain', 'aeroplane', 'water', 'face', 'car',
                'bicycle', 'flower', 'sign', 'bird', 'book', 'chair', 'road',
                'cat', 'dog', 'body', 'boat'])
        images = glob(os.path.join(self.directory, "Images", "*.bmp"))
        self.images = [os.path.basename(f)[:-4] for f in images]
        #self.ignored_classes = map(self.classes.index, ['horse', 'mountain'])
        self.n_images = len(self.images)
        self.n_classes = len(self.classes)  # - len(self.ignored_classes)

        if self.n_images == 0:
            raise ValueError("no images found in directory %s", self.directory)

        self.colors = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0],
            [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
            [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192,
                128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128,
                64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
            [192, 64, 0]])
        self.convert = [99. / 1000,  587. / 1000,  114. / 1000]
        self.label_dict = np.dot(self.colors, self.convert).tolist()

    def get_images(self):
        return [self.get_image(image) for image in self.images]

    def get_image(self, image):
        return np.array(Image.open(os.path.join(self.directory, "Images", "%s.bmp" % image)))

    def get_ground_truth(self, image):
        img = Image.open(os.path.join(self.directory, "GroundTruth", "%s_GT.bmp" % image))
        img = np.array(img)
        dp = np.dot(img, self.convert)
        labels = np.unique(dp)  # find all occuring labels
        img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        for l in labels:
            val = self.label_dict.index(l)
            img[dp == l] = val
        return img

if __name__ == "__main__":
    msrc = MSRCDataset("/home/local/datasets/MSRC_ObjCategImageDatabase_v2/")
    from IPython import embed
    embed()
