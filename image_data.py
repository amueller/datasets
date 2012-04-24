import os
import numpy as np
from scipy.io import loadmat
import pdist2


class ImageData(object):
    def __init__(self, directory, filename, dataset=None):
        self.filename = filename
        self.directory = directory
        self.dataset = dataset

        #self.get_segment_features_normalized = \
                #memory.cache(self.get_segment_features_normalized)
        #self.get_segment_masks = \
                #memory.cache(self.get_segment_masks)
        #self.get_ground_truth = memory.cache(self.get_ground_truth)
        #self.labels = memory.cache(self.labels)
        #self.num_segments = memory.cache(self.num_segments)

    def get_segment_features(self,feat):
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

    def get_segment_features_normalized(self, feat):
        features = self.get_segment_features(feat)
        normalized_features = pdist2.normalize(features)
        return normalized_features

    def get_segment_masks(self):
        print(os.path.join(self.directory, 'masks/%s.mat'%self.filename))
        return loadmat(os.path.join(self.directory, 'masks/%s.mat'%self.filename))['masks']

    def get_data(self):
        import matplotlib.pyplot as plt
        image = plt.imread(os.path.join(self.directory,self.filename+'.png'))
        return image[::-1,:,:] # for some reason we have to flip it...

    def get_ground_truth(self):
        import matplotlib.pyplot as plt
        image = plt.imread(os.path.join(self.directory,self.filename+'_GT.bmp'))
        return image[::-1,:,:] # for some reason we have to flip it...


    #def labels(self):
        #import matplotlib.pyplot as plt
        #data = plt.imread(os.path.join(self.directory,self.filename+'_GT.bmp'))
        #labels=map(label_dict.index,np.unique(np.dot(data,convert))[1:]) #only msrc [1:] get's rid of 0'
        #return labels

    def num_segments(self):
        return self.get_segment_masks().shape[2]



class RankedImageData(ImageData):
    def score_threshold(self):
        if hasattr(self,"_score_threshold"):
            return self._score_threshold
        else:
            #self._score_threshold=sorted(self.scores.tolist())[-50]
            max_segments = min(100,self.scores().shape[0])
            #max_segments = self.scores.shape[0]
            self._score_threshold=sorted(self.scores().tolist())[-max_segments]
            return self._score_threshold

    def ranking(self):
        max_segments = min(100,self.scores().shape[0])
        #max_segments = self.scores().shape[0]
        return np.argsort(self.scores())[::-1][:max_segments] # use original ranking!
        #ranking_path = os.path.join(self.directory,"scores","reranked_%s.npy"%self.filename)
        #if os.path.exists(ranking_path):
            #ranking = np.load(ranking_path)
        #else:
            #ranking = diversify(self.scores(),self.get_segment_masks())[:max_segments]
            #np.save(ranking_path,ranking)
        #return ranking

    def get_segment_features(self,feat):
        features = super(RankedImageData, self).get_segment_features(feat)
        return features[:, self.scores() >= self.score_threshold()]

    def scores(self):
        scores = np.squeeze(loadmat(os.path.join(self.directory,'scores/%s'%(self.filename)))['scores'].astype("float32"))
        return scores

    def get_segment_features_normalized(self, feat):
        features = self.get_segment_features(feat)[:,self.ranking()]
        #normalized_features = pdist2.normalize(features[:,self.ranking()].copy('F'))
        normalized_features = pdist2.normalize(features)
        return normalized_features

    def get_segment_masks(self):
        return loadmat(os.path.join(self.directory, 'masks/%s.mat'%self.filename))['masks'][:,:,self.scores()>=self.score_threshold()]
