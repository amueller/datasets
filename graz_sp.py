import os
import numpy as np

from scipy.io.matlab import loadmat

from graz import GrazImageData, GrazSegmentDataset


class GrazImageDataSP(GrazImageData):
    def num_segments(self):
        return len(np.unique(self.get_superpixels()))
    def ground_truth_on_sp(self):
        numpy_file = os.path.join(self.directory,'superpixels/gt/%s.npy'%self.filename)
        if os.path.exists(numpy_file):
            votes = np.load(numpy_file)
        else:
            sp = self.get_superpixels()
            gt = self.get_ground_truth()
            gt_vote = map(lambda x: np.histogram(gt[sp==x],bins=np.arange(5))[0], np.unique(sp))
            gt_vote = np.array(gt_vote).astype(np.float)
            votes = np.argmax(gt_vote, axis=1)
            if not os.path.exists(os.path.dirname(numpy_file)):
                  os.makedirs(os.path.dirname(numpy_file))
            np.save(numpy_file, votes)
            #gt_vote /= np.sum(gt_vote, axis=1)[:, np.newaxis]
            #votes = gt_vote[sp,:]
            #votes = np.dstack([votes[:,:,1:], gt[:, :, np.newaxis]])
        return votes

    def get_segment_features_normalized(self, feat):
        return self.sift_on_superpixels(feat)

    def sift_on_superpixels(self, feat):
        codebook = self.dataset.codebooks[feat]
        numpy_file = os.path.join(self.directory,'superpixel_descriptors/%s_%s.npy'%(feat,self.filename))
        if os.path.exists(numpy_file):
            hists = np.load(numpy_file)
        else:
            from scipy.cluster.vq import vq
            sp = self.get_superpixels()
            matlab_dict = loadmat(os.path.join(self.directory,'MyMeasurements/%s/%s'%(feat,self.filename)))
            locations = matlab_dict['F'][:2,:].astype(np.int)
            descriptors = matlab_dict['D']
            descriptor_words = vq(descriptors.T, codebook.T)[0]
            desc_to_sp = sp[locations[1,:],locations[0,:]]
            hists = np.zeros((len(np.unique(sp)), codebook.shape[1]))
            for p, desc in zip(desc_to_sp, descriptor_words):
                hists[p, desc] +=1
            hists /= (hists.sum(axis=1)[:, np.newaxis] + 1e-10)
            if not os.path.exists(os.path.dirname(numpy_file)):
                  os.makedirs(os.path.dirname(numpy_file))
            np.save(numpy_file, hists)
        return hists

    def get_superpixels(self):
        numpy_file = os.path.join(self.directory,'superpixels/%s.npy'%self.filename)
        if os.path.exists(numpy_file):
            superpixels = np.load(numpy_file)
        else:
            from quickshift import quickshift
            # most results: 10, 6
            # new rbf results: 20, 6
            # 2nd rbf: 20, 10
            # 3rd rbf: 10, 10
            # 4th rbf: 10, 20
            # 5th rbf: tau=8 sigma=2 
            superpixels = quickshift((self.get_data()*255).astype(np.ubyte), 10, 10)
            superpixels = superpixels[1][::-1, :]
            labels = np.unique(superpixels)
            new_labels = np.arange(len(labels))
            mapping = dict(zip(labels, new_labels))
            sp_new = map(lambda x: mapping[x],superpixels.ravel())
            superpixels = np.array(sp_new).reshape(superpixels.shape)
            if not os.path.exists(os.path.dirname(numpy_file)):
                  os.makedirs(os.path.dirname(numpy_file))
            np.save(numpy_file, superpixels)
        return superpixels

class GrazSegmentDatasetSP(GrazSegmentDataset):
    def __init__(self, directory, filename=None):
        self.images = []
        self.directory = directory
        if filename != None:
            imagefiles = np.loadtxt(os.path.join(directory, filename), np.str)
            for f in imagefiles:
                self.images.append(GrazImageDataSP(directory,"%s.image"%f, self))
        #self.features = ['dense_color_sift_3_scales', 'dense_sift_4_scales']
        self.features = ['dense_sift_4_scales']
        self.codebooks = dict()
        self.codebooks['dense_sift_4_scales'] = loadmat(os.path.join(self.directory, "MyCodebooks/kmeans_dense_sift_4_scales_300_words.mat"))['codebook']
        self.codebooks['dense_color_sift_3_scales'] = loadmat(os.path.join(self.directory, "MyCodebooks/kmeans_dense_color_sift_3_scales_300_words.mat"))['codebook']
        self.num_features = len(self.features)
        self.images = self.images
        self.classes = ["cars","bikes","people"]
        self.train_classes = np.array([0,1,2]) # no sky and gras
        self.num_classes = len(self.train_classes)
