import numpy as np
import os
from joblib import Memory
from image_data import ImageData
#from rerank_masks import diversify

memory = Memory(cachedir="cache", verbose=0)

voc_directory = '../pascal/'


#def histcode(feat,codebook):
#    from scipy.cluster.vq import vq
#    codes = vq(feat.T,codebook.T)[0]
#    bins = np.bincount(codes)
#    return bins/float(np.sum(bins))

#def compute_feature_on_masks(image,feature):
#    eval("import feature_extraction")
#    feature_func = eval("feature_extraction.%s.%s"%(feature,feature))
#    numpy_name =os.path.join(image.directory,'python_features/mask_%s/%s.npy'%(feature, image.filename))
#    from joblib import Parallel, delayed
#    if not os.path.exists(numpy_name):
#        im_data = image.get_data()
#        masks = image.get_segment_masks()
#        #for mask in np.split(masks,masks.shape[2],axis=2):
#            #desc = phog(image*mask,cellsize=32)
#            #descs.append(desc)
#        descs = Parallel(n_jobs=-1)(delayed(feature_func)(im_data*mask,windowsize=32) for mask in np.split(masks,masks.shape[2],axis=2))
#        descs = np.array(descs)
#        if not os.path.exists(os.path.dirname(numpy_name)):
#              os.makedirs(os.path.dirname(numpy_name))
#        np.save(numpy_name,descs)
#    else:
#        descs = np.load(numpy_name)
#    return descs



class SegmentDataset(object):
    def __init__(self, directory, filename = None):
        self.images = []
        self.features = ['dummy_masks_bow_dense_color_sift_3_scales_figure_300/', 'dummy_masks_bow_dense_sift_4_scales_figure_300', 'dummy_masks_mask_phog_scale_inv_20_orientations_2_levels']
        self.num_features = len(self.features)
        if filename!=None:
            imagefiles = np.loadtxt(os.path.join(directory, filename), np.str)
            for f in imagefiles:
                #print("Filename %s"%f)
                self.images.append(ImageData(directory,f))
            self.images = np.array(self.images)
        #train_classes = np.array([0,2,3,8,10,11,12]) # no sky and gras

    def get_binary_labels(self, train_class):
        labels = [train_class in image.labels() for image in self.images]
        labels = (2*np.array(labels) - 1 ).astype('float')
        return labels

    def get_1vs1_dataset(self, train_class1, train_class2):
        data_1on1 = self.__class__(self.directory)
        data_1on1.images = filter(lambda x: (train_class1 in x.labels()) ^ (train_class2 in x.labels()),
            self.images) # exclusive or, otherwise i don't know how to train ^^

        # 1 and -1 labels
        labels_1on1 = np.array([2 * (train_class1 in image.labels()) - 1 for image in data_1on1.images],np.float)
        return data_1on1, labels_1on1

