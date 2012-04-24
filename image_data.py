import os
import numpy as np
from scipy.io import loadmat

class ImageDataset(object):
    def __init__(self, directory):
        self.directory = directory
        self.classes = []

    def get_images(self,feat):
        numpy_name =os.path.join(self.directory,'python_features/%s/%s.npy'%(feat,self.filename))
        if os.path.exists(numpy_name):
            features = np.load(numpy_name)
            features = features.reshape(features.shape[0],-1)
        else:
            print("loadmat")
            new_feat =loadmat(os.path.join(self.directory,'MyMeasurements/%s/%s'%(feat,self.filename)))['D'].astype("float32")
            if not os.path.exists(os.path.dirname(numpy_name)):
                  os.makedirs(os.path.dirname(numpy_name))
            np.save(numpy_name,new_feat)
            features = new_feat
        return features

    def get_features(self, feat):
        pass


    def get_superpixels(self):
        pass

    def get_ground_truth(self):
        pass
