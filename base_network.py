
import matplotlib
matplotlib.use("Agg")


import keras
import cv2
import os
import tensorflow 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adamax
from keras.models import Model


def get_base(base_name, img_size, n_classes):
    
    if base_name == "xception":
        base = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
       
    if base_name == "vgg16":
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "vgg19":
        base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
       
    if base_name == "resnet50v2":
        base = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "resnet101":
        base = tf.keras.applications.ResNet101(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "resnet101v2":
        base = tf.keras.applications.ResNet101V2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "resnet152":
        base = tf.keras.applications.ResNet152(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
       
    if base_name == "resnet152v2":
        base = tf.keras.applications.ResNet152V2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "inceptionv3":
        base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "inceptionrestnetv2":
        base = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
    
    if base_name == "mobilenet":
        base = tf.keras.applications.MobileNet(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
       
    if base_name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "densenet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
    
    if base_name == "densenet201":
        base = tf.keras.applications.densenet201(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb1":
        base = tf.keras.applications.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb2":
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb3":
        base = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb4":
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb5":
        base = tf.keras.applications.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb6":
        base = tf.keras.applications.EfficientNetB6(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "efficientnetb7":
        base = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
    
    x = base.output
    x = Flatten()(x)
    
    outs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base.inputs, outputs= outs)
    return model