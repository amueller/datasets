import os
from glob import glob

import numpy as np
import Image

#from image_data import ImageDataset


#def generate_val_train_msrc(directory, random_init):
#    import random
#    with open(os.path.join(directory, 'images.txt')) as f:
#        image_names = f.readlines()
#    class_images = [[] for i in xrange(1, 9)]
#    for image_name in image_names:
#        class_images[int(image_name[0]) - 1].append(image_name)
#    train_file = os.path.join(directory, 'train_%d.txt' % random_init)
#    with open(, 'w') as train_list:
#        val_file = os.path.join(directory, 'val_%d.txt' % random_init)
#        with open(val_file, 'w') as val_list:
#            for class_list in class_images:
#                random.shuffle(class_list)
#                train_list.writelines(class_list[:len(class_list) / 2])
#                val_list.writelines(class_list[len(class_list) / 2:])


class MSRCDataset(object):
    def __init__(self,  directory=None, rm_mountain_horse=True):
        if directory is None:
            directory = "/home/local/datasets/MSRC_ObjCategImageDatabase_v2/"
        self.directory = directory
        classes = ['void', 'building', 'grass', 'tree', 'cow',
            'horse', 'sheep', 'sky', 'mountain', 'aeroplane', 'water', 'face',
            'car', 'bicycle', 'flower', 'sign', 'bird', 'book', 'chair',
            'road', 'cat', 'dog', 'body', 'boat']
        images = glob(os.path.join(self.directory, "Images", "*.bmp"))
        self.images = [os.path.basename(f)[:-4] for f in images]
        self.n_images = len(self.images)

        if self.n_images == 0:
            raise ValueError("no images found in directory %s", self.directory)

        colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
            [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
            [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192,
                128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128,
                64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
            [192, 64, 0]]
        self.convert = [99. / 1000,  587. / 1000,  114. / 1000]
        converted_colors = np.dot(np.array(colors), self.convert).tolist()
        label_dict = dict()
        if rm_mountain_horse:
            horse_idx = classes.index("horse")
            mountain_idx = classes.index("mountain")
            horse_color = converted_colors[horse_idx]
            mountain_color = converted_colors[mountain_idx]
            label_dict[horse_color] = 0
            label_dict[mountain_color] = 0
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

if __name__ == "__main__":
    msrc = MSRCDataset()
    from IPython import embed
    embed()
