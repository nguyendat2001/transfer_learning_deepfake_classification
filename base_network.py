
import matplotlib
matplotlib.use("Agg")


import keras
import cv2
import os
import tensorflow 
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten, MaxPool2D
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda
from keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def entry_flow(inputs):
    
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for size in [128, 256, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(  # Project residual
            size, 1, strides=2, padding='same')(previous_block_activation)           
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    return x

def middle_flow(x, num_blocks=8):
      
    previous_block_activation = x

    for _ in range(num_blocks):
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, previous_block_activation])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    return x

def exit_flow(x, num_classes=1000):
      
    previous_block_activation = x

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    residual = layers.Conv2D(  # Project residual
      1024, 1, strides=2, padding='same')(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual

    x = layers.SeparableConv2D(1536, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    # if num_classes == 1:
    #     activation = 'sigmoid'
    # else:
    #     activation = 'softmax'
    return x

def vgg16(img_size):
    model = Sequential()
    model.add(Conv2D(input_shape=(img_size,img_size,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    # model.add(Dense(units=n_class, activation="softmax"))
    return model

def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
        
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
        
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
        
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
        
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
   
    return X

def ResNet50(input_shape = (50, 50, 3)):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)
    X = Flatten()(X)
    model = Model(inputs = X_input, outputs = X)
    return model

def get_base(base_name, img_size, n_classes):
    
    if base_name == "xception":
        inputs = keras.Input(shape=(img_size, img_size, 3))
        outputs = exit_flow(middle_flow(entry_flow(inputs)))
        base = keras.Model(inputs, outputs)
        # base = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
       
    if base_name == "vgg16":
        # base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        base = vgg16(img_size)
        
    if base_name == "vgg19":
        base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
    if base_name == "resnet50":
        base = ResNet50((img_size,img_size,3))
       
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
        base = tf.keras.applications.Densenet201(include_top=False, weights="imagenet", input_shape=(img_size,img_size,3) )
        
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