import argparse
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
# from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pickle
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import keras
import cv2
import os
import json
import tensorflow 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
import base_network
# !pip install mediapipe
# import mediapipe as mp

n_class = 2

def read_data(path):
    def load_data(path):
    file = open(path, 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np_utils.to_categorical(labels, n_class)
    pixels = np.array(pixels)
    labels = np.array(labels)
    print(pixels.shape)
    print(labels.shape)

    return pixels, labels


def train_val(exp_dict , save_dir, data_dir):
    figures_path = save_dir
    
    X,y = read_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = base_network.get_base(exp_dict['base'],exp_dict,2)
    model.
    
    BS = exp_dict['batch_size']
    
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, brightness_range=[1,1.5],
        horizontal_flip=True, fill_mode="nearest")
    
    opt = Adam(lr=0.001)
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy","AUC"])
    
    checkpoint_path = os.path.join(figures_path,"best_model.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
#     earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
#                                                   patience=15)
    
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    history = model.fit(X_train,y_train,
        validation_data=(X_test,y_test,BS), 
        epochs=EPOCHS, batch_size = BS,
        callbacks=[checkpoint])

    # eval
    model.save(os.path.join(figures_path,'model.h5'))
    
    hist_df = pd.DataFrame(history.history)
    name_history = os.path.join(figures_path,'history.csv')
    with open(name_history, mode='w') as f:
        hist_df.to_csv(f)
        
#     loss_test,accuracy_test, auc_test = model.evaluate(X_test, y_test, batch_size=exp_dict['batch_size'])
#     print('LOSS TEST: ', loss_test)
#     print("ACCURACY TEST: ", accuracy_test)
#     print('AUC TEST: ', auc_test)

#     supervisor = 'LOSS TEST: '+ str(loss_test)+"\nACCURACY TEST: "+ str(accuracy_test)+'\nauc test: '+ str(auc_test)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--savedir_base', default='CovidSeg/save')
    parser.add_argument('-d', '--datadir', default='CovidSeg/dataset')
    parser.add_argument("-b", "--base", default='') # efficientnet-b0 
    parser.add_argument("-bs", "--batch_size", type=int, default=2) # batch_size
    parser.add_argument("-i", "--im_size", type=int, default=512) # image size for input
    parser.add_argument("-e", "--epochs", type=int, default=512)
    parser.add_argument('-o', '--opt', default='adam') # optimizer adam or SGD 
#     parser.add_argument('-ag', '--augmentation', type=bool, default=True) # optimizer adam or SGD 
    args = parser.parse_args()
    exp_dict = []
    exp_dict['base'] = args.base
    exp_dict['img_size'] = args.im_size
    exp_dict['batch_size'] = args.batch_size
    exp_dict['optimizer'] = args.opt
    exp_dict['epochs'] = args.epochs
    
    with open('exp_dict.json', 'w', encoding='utf-8') as f:
        json.dump(exp_dict, f, ensure_ascii=False, indent=4)