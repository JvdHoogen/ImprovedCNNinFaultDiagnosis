import pandas as pd
import os
import random as rn
import numpy as np
import keras
import tensorflow as tf
from tensorflow import set_random_seed
from keras import backend as k
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, Activation, AveragePooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras import models
from sklearn import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate
import multivariate_cwru



os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for random number generators
rn.seed(1254)
np.random.seed(1)
set_random_seed(2)

# Configure processing allocation to only one core for reproducible results
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(),config=config)
k.set_session(sess)

