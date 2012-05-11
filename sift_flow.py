import os

import numpy as np
from scipy.io import loadmat
from glob import glob

class SIFTFlow(object):
    def __init__(self, directory):
        self.directory = directory

    #def get_ground_truth(self):
        #segment_dir = os.path.join("LMsegments", "spatial_envelope_256x256_static_8outdoorcategories")
        #with filename in glob(os.path.join(segment_dir, "*.mat"):




