import os
import numpy as np
from get_features import ImageData,SegmentDataset
from joblib import Memory
memory = Memory(cachedir="cache", verbose=0)

class GrazImageData(ImageData):
    def __init__(self, directory, filename, dataset=None):
        self.classes = ["cars","bikes","people"]
        super(GrazImageData, self).__init__(directory, filename, dataset)

    def get_data(self):
        import matplotlib.pyplot as plt
        image = plt.imread(os.path.join(self.directory,self.class_string(),self.filename+'.png'))
        return image

    def get_ground_truth(self):
        import matplotlib.pyplot as plt
        from glob import glob
        image = None
        for masks in glob(os.path.join(self.directory,self.class_string(),self.filename[:-5] + 'mask.*')):
            if image==None:
                image = plt.imread(masks)[:,:,0] # only red channel
            else:
                image += plt.imread(masks)[:,:,0] # only red channel
        return (image!=0)*(self.class_num()+1)

    def labels(self):
        return [self.class_num()]

    def class_num(self):
        my_class_string = self.filename[:self.filename.find("_")]
        if my_class_string=="carsgraz":
            return 0
        elif my_class_string=="bike":
            return 1
        elif my_class_string=="person":
            return 2
        else:
            raise ValueError()

    def class_string(self):
        return self.classes[self.class_num()]

class GrazSegmentDataset(SegmentDataset):
    def __init__(self, directory, filename=None):
        self.images = []
        self.directory = directory
        if filename != None:
            imagefiles = np.loadtxt(os.path.join(directory, filename), np.str)
            for f in imagefiles:
                self.images.append(GrazImageData(directory,"%s.image"%f))
        #self.features = ['dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']
        #self.features = ['mask_phog']
        #self.features = ['mask_lbp']
        self.features = ['dummy_masks_bow_dense_color_sift_3_scales_figure_300/', 'dummy_masks_bow_dense_sift_4_scales_figure_300', 'dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']
        #self.features = ['dummy_masks_bow_dense_sift_4_scales_figure_300']
        self.num_features = len(self.features)
        #for feat in ['dummy_masks_bow_dense_color_sift_3_scales_figure_300/', 'dummy_masks_bow_dense_sift_4_scales_figure_300', 'dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']:
        self.images = self.images
        self.classes = ["cars","bikes","people"]
        self.train_classes = np.array([0,1,2]) # no sky and gras
        self.num_classes = len(self.train_classes)

    def get_binary_labels(self, train_class):
        labels = [image.class_num() == train_class for image in self.images]
        labels = (2*np.array(labels) - 1 ).astype('float')
        return labels
