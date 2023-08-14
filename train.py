import argparse
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
# from model.livenessnet import LivenessNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pickle
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

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

n_class = 4

# def read_data(path):
#     def load_data(path):
#     file = open(path, 'rb')

#     # dump information to that file
#     (pixels, labels) = pickle.load(file)

#     # close the file
#     file.close()
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)
#     labels = np_utils.to_categorical(labels, n_class)
#     pixels = np.array(pixels)
#     labels = np.array(labels)
#     print(pixels.shape)
#     print(labels.shape)

#     return pixels, labels


def train_val(args , save_dir):
    
    # train_datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     validation_split=0.2)
    # valid_datagen = ImageDataGenerator(rescale=1./255)
        
    figures_path = save_dir
    
    model = base_network.get_base(args.base,args.im_size,5)
    model.summary()
    # model.
    
    BS = args.batch_size
    
    aug = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, brightness_range=[1,1.5],
        horizontal_flip=True, fill_mode="nearest")
    
    aug_tmp = ImageDataGenerator(rescale=1./255)
    
    train_generator = aug.flow_from_directory(directory=args.train, 
                                                    target_size=(args.im_size, args.im_size),
                                                    # classes=['NORMAL','PNEUMONIA','TURBERCULOSIS'],
                                                    # color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=1234)
    
    val_generator = aug_tmp.flow_from_directory(directory=args.val,
                                                    target_size=(args.im_size, args.im_size),
                                                    # classes=['NORMAL','PNEUMONIA','TURBERCULOSIS'],
                                                    # color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=1234)
    
    test_generator = aug_tmp.flow_from_directory(directory=args.test,
                                                    target_size=(args.im_size, args.im_size),
                                                    # classes=['NORMAL','PNEUMONIA','TURBERCULOSIS'],
                                                    # color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=1234)
    
    opt = Adam(lr=0.001)
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy","AUC",f1_m,precision_m, recall_m])
    
    checkpoint_path = os.path.join(figures_path,"best_model.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
#     earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
#                                                   patience=15)
    
    step_train = train_generator.n//64
    step_val = val_generator.n//64
    
    print("[INFO] training network for {} epochs...".format(args.epochs ))
    # history = model.fit(train_generator,
    #     validation_data=val_generator, 
    #     epochs=args.epochs,
    #     callbacks=[checkpoint])
    history = model.fit_generator(generator=train_generator, steps_per_epoch=step_train,
                    validation_data=val_generator,
                    validation_steps=step_val,
                    callbacks=[checkpoint],
                    epochs=args.epochs)

    # eval
    model.save(os.path.join(figures_path,'model.h5'))
    
    model.load_weights(checkpoint_path)
    
    hist_df = pd.DataFrame(history.history)
    name_history = os.path.join(figures_path,'history.csv')
    with open(name_history, mode='w') as f:
        hist_df.to_csv(f)
    
    summaries = []
    
    loss_test,accuracy_test,auc_test ,f1_test,precision_test, recall_test = model.evaluate(test_generator)
    print('LOSS TEST: ', loss_test)
    print("ACCURACY TEST: ", accuracy_test)
    print('AUC TEST: ', auc_test)
    summaries.append([loss_test, accuracy_test, auc_test, f1_test, precision_test, recall_test])

    pd.DataFrame(summaries).to_csv(os.path.join(figures_path, 'result.csv')) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--savedir_base', default='CovidSeg/save')
    parser.add_argument('-tr', '--train', default='CovidSeg/dataset')
    parser.add_argument('-te', '--test', default='CovidSeg/dataset')
    parser.add_argument('-va', '--val', default='CovidSeg/dataset')
    parser.add_argument("-b", "--base", default='') # efficientnet-b0 
    parser.add_argument("-bs", "--batch_size", type=int, default=64) # batch_size
    parser.add_argument("-i", "--im_size", type=int, default=256) # image size for input
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument('-o', '--opt', default='adam') # optimizer adam or SGD 
#     parser.add_argument('-ag', '--augmentation', type=bool, default=True) # optimizer adam or SGD 
    args = parser.parse_args()
    # exp_dict = []
    # exp_dict['base'] = args.base
    # exp_dict['img_size'] = args.im_size
    # exp_dict['batch_size'] = args.batch_size
    # exp_dict['optimizer'] = args.opt
    # exp_dict['epochs'] = args.epochs
    # exp_dict['train'] = args.train
    # exp_dict['test'] = args.test
    # exp_dict['val'] = args.val
    
    # with open('exp_dict.json', 'w', encoding='utf-8') as f:
    #     json.dump(exp_dict, f, ensure_ascii=False, indent=4)
        
    train_val(args , args.savedir_base)