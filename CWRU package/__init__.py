import os,re
import glob
import errno
import random
import urllib.request 
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
import sys
    
# Filtering values of the two sensors    
def keyfilter(dictionary_keys):
    keylist = []
    for key in dictionary_keys:
        if "DE_time" in str(key):
            keylist.append(key)
        if "FE_time" in str(key):
            keylist.append(key)
    return keylist[0],keylist[1]

 
class CWRU:
    # Exp = experiment name, rpm = rotations per minute, length = length of sequence, 
    # Trainsplit = division of train and testsplit, seed = seed for random shuffle
    def __init__(self, exp, length, trainsplit, seed, *rpm):
        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print("wrong experiment name: {}".format(exp))
            sys.exit(1) 
        for i in rpm:
            if i not in ('1797', '1772', '1750', '1730'): 
                print("wrong rpm value: {}".format(rpm))
                sys.exit(1)
        # Root directory of all data and loading in text file
        rdir = os.path.join(os.path.expanduser('~'), 'Datasets/CWRU')
        cur_path = os.path.dirname(__file__)
        fmeta = os.path.join(cur_path, "metadata.txt")

        # Read text file and load all separate http addresses
        all_lines = open(fmeta).readlines() 
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] in exp or l[0] == 'NormalBaseline') and l[1] in rpm:
                lines.append(l)
 
        self.length = length  # sequence length
        self.seed = seed
        self.trainsplit = trainsplit
        self._load_and_slice_data(rdir, lines)
        self._shuffle() # shuffle training and test arrays
        self.labels = tuple(line[2] for line in lines) # Label names 
        self.nclasses = len(self.labels)  # Number of classes
 
    # Create directories for the download files to store
    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)
 
    # Download files from corresponding HTML addresses from 'metadata.txt'
    def _download(self, fpath, link):
        print("Downloading to: '{}'".format(fpath))
        urllib.request.urlretrieve(link, fpath)
        
    # Extract data from .mat files and preprocess them into sliced datasets
    def _load_and_slice_data(self, rdir, infos):
        self.X_train = np.zeros((0, self.length, 2))
        self.X_test = np.zeros((0, self.length, 2))
        self.y_train = []
        self.y_test = []
        for idx, info in enumerate(infos):
            # Directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            if not os.path.exists(fpath):
                self._download(fpath, info[3])
            
            # Load in files and combine into one time series
            mat_dict = loadmat(fpath)
            key1,key2 = keyfilter(mat_dict.keys())
            time_series = np.hstack((mat_dict[key1],mat_dict[key2]))

            # Remove leftover datapoints based on sequence length
            idx_last = -(time_series.shape[0] % self.length)
            if idx_last < 0:    
                clips = time_series[:idx_last].reshape(-1, self.length,2)
            else:
                clips = time_series[idx_last:].reshape(-1, self.length,2)

            # Partition train and test set in separate arrays
            n = clips.shape[0]
            n_split = int(self.trainsplit * n)
            self.X_train = np.vstack((self.X_train, clips[:n_split]))
            self.X_test = np.vstack((self.X_test, clips[n_split:]))
            self.y_train += [idx] * n_split
            self.y_test += [idx] * (clips.shape[0] - n_split)

    def _shuffle(self):
        # Shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(self.seed).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))
 
        # Shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(self.seed).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))
