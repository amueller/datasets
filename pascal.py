import os
import numpy as np
import matplotlib.colors
import Image
from image_data import RankedImageData
from segment_dataset import SegmentDataset
from joblib import Memory
memory = Memory(cachedir="cache", verbose=0)

#from IPython.core.debugger import Tracer
#tracer = Tracer()

class PascalImageData(RankedImageData):
    def __init__(self, directory, filename, dataset=None):
        super(PascalImageData, self).__init__(directory, filename, dataset)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                'sofa', 'train', 'tvmonitor']
        self.cmap = None # colormap for classes. not hardcoded
        self._classes_present = None
        self.segment_gt_overlap = memory.cache(self.segment_gt_overlap)
        self.classes_present = memory.cache(self.classes_present)

    def get_data(self):
        image = np.array(Image.open(os.path.join(self.directory, "JPEGImages", self.filename+'.jpg')))
        return image

    def segment_gt_overlap(self, classnum, return_ratio = True):
        segs = self.get_segment_masks().astype(np.bool)
        gt = self.get_ground_truth()
        gt_class = (gt==classnum)[:,:, np.newaxis]
        gt_dc = (gt == 255)[:,:, np.newaxis]
        intersection = segs * gt_class*(1-gt_dc)
        union = (segs + gt_class)*(1-gt_dc)
        int_sum = intersection.sum(axis=0).sum(axis=0)
        union_sum = union.sum(axis=0).sum(axis=0).astype(np.float)
        if return_ratio:
            return int_sum / union_sum
        return np.array([int_sum, union_sum])

    def get_ground_truth(self):
        mask_file = os.path.join(self.directory, "SegmentationClass", self.filename + '.png')
        mask = Image.open(mask_file)
        palette = mask.getpalette()
        # generate matplotlib colormap
        self.cmap = matplotlib.colors.ListedColormap(np.array(palette).reshape(256,3)/255.)
        mask_array = np.array(mask.getdata()).reshape(mask.size[1], mask.size[0])
        return mask_array

    def labels(self):
        return [self.class_num()]

    def classes_present(self):
        if self._classes_present == None:
            self._classes_present = np.unique(self.get_ground_truth())

        return self._classes_present

class PascalSegmentDataset(SegmentDataset):
    def __init__(self, directory, filename=None):
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                'sofa', 'train', 'tvmonitor']
        self.images = []
        self.directory = directory
        if filename != None:
            imagefiles = np.loadtxt(os.path.join(directory, filename), np.str)
            for f in imagefiles:
                self.images.append(PascalImageData(directory,"%s"%f))
        self.features = ['dummy_masks_bow_dense_color_sift_3_scales_figure_300/', 'dummy_masks_bow_dense_sift_4_scales_figure_300', 'dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']
        #self.features = ['dummy_masks_bow_dense_sift_4_scales_figure_300']
        self.num_features = len(self.features)
        #for feat in ['dummy_masks_bow_dense_color_sift_3_scales_figure_300/', 'dummy_masks_bow_dense_sift_4_scales_figure_300', 'dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']:
        self.images = self.images
        self.num_classes = len(self.classes)

    def get_binary_labels(self, train_class):
        labels = [image.class_num() == train_class for image in self.images]
        labels = (2*np.array(labels) - 1 ).astype('float')
        return labels
