from unicodedata import name
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Dense, AveragePooling2D, Dropout, Flatten
from tensorflow_examples.models.pix2pix import pix2pix
import time
from CustomDataGen import CustomDataGen
import random
from datetime import datetime
import os
import cv2
import math
import logging
from sklearn.utils import class_weight
from Model_Optimizer import CosineAnnealingScheduler  # CustomSchedule
from tensorflow.keras.utils import Progbar
#from models_development import  DualResNet50, DualAttResNet50, Crossview2DualAttResNet50
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import backend as K
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from Special_layers import Patches, PatchEncoder
import tensorflow_addons as tfa
from tensorflow.keras import layers


class Create_Model():
    def __init__(self, args):
        self.args = args

    def Averaged_Excitation_Attention(self, x, y, t=0):
        x_att = GlobalAveragePooling2D()(x)
        #x_att = GlobalMaxPooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        Combined_att = tf.keras.layers.Average()([x_d, y_d])

        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])
        if t == 1:
            return x_att, y_att, Combined_att
        else:
            return x_att, y_att
        
    def Maximum_Excitation_Attention(self, x, y, t=0):
        x_att = GlobalAveragePooling2D()(x)
        #x_att = GlobalMaxPooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        #x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        #y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        Combined_att =  concatenate([x_d, y_d])
        Combined_att = Dense(y_att.shape[1], activation='sigmoid')(Combined_att)
        

        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])
        if t == 1:
            return x_att, y_att, Combined_att
        else:
            return x_att, y_att

    def Averaged_Excitation_Attention_skip(self, x, y, a, t=0):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        Combined_att = tf.keras.layers.Average()([x_d, y_d])
        if t == 1:
            Combined_att = Add()([Combined_att, a])

        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])

        return x_att, y_att

    def Self_Excitation_Attention(self, x, y):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        x_att = tf.keras.layers.Multiply()([x, x_d])
        y_att = tf.keras.layers.Multiply()([y, y_d])
        return x_att, y_att

    def Addition_Excitation_Attention(self, x, y, t=0):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        Combined_att = Add()([x_d, y_d])

        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])
        if t == 1:
            return x_att, y_att, Combined_att
        else:
            return x_att, y_att

    def Addition_Excitation_Attention_skip(self, x, y, a, t=0):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16)(x_att)
        x_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(x_d)
        x_d = Activation('relu')(x_d)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)

        y_att = GlobalAveragePooling2D()(y)
        #y_att = GlobalMaxPooling2D()(y)
        y_d = Dense(y_att.shape[1]/16)(y_att)
        y_d = BatchNormalization(momentum=0.1, epsilon=0.0001)(y_d)
        y_d = Activation('relu')(y_d)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)

        Combined_att = Add()([x_d, y_d])
        if t == 1:
            Combined_att = Add()([Combined_att, a])

        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])

        return x_att, y_att

    def Pretrained_Resnet50(self):
        baseA = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1")
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        POOLB = MaxPooling2D((7, 7))(baseB.output)
        fc2 = self.Output_Block(POOLA, POOLB)
        model = Model(inputs=[baseA.input, baseB.input], outputs=fc2)
        return model
    
    def Pretrained_EfficientNetB0(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1_1")
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        POOLB = MaxPooling2D((7, 7))(baseB.output)
        fc2 = self.Output_Block(POOLA, POOLB)
        model = Model(inputs=[baseA.input, baseB.input], outputs=fc2)
        return model
    
    def Pretrained_EfficientNetB4(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1_1")
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        POOLB = MaxPooling2D((7, 7))(baseB.output)
        fc2 = self.Output_Block(POOLA, POOLB)
        model = Model(inputs=[baseA.input, baseB.input], outputs=fc2)
        return model
    
    def Pretrained_EfficientNetB4_single(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000, drop_connect_rate=0.4)
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        dropout = Dropout(self.args.dropout)(POOLA)
        flattened = Flatten()(dropout)
        fc1 = Dense(self.args.img_size*2,
                    activation="relu")(flattened)  # was 100
        fc2 = Dense(self.args.num_output_classes,activation="softmax")(fc1)
        model = Model(inputs=[baseA.input], outputs=fc2)
        return model
        
        

    def Pretrained_Model_Block_Resnet50(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        new_layer_input_name = 'conv4_block6_out'
        baseA = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1")
        for lay in range(0, len(baseB.layers)):
            if baseB.layers[lay].name == new_layer_input_name:
                step = lay
        TOPA = Model(inputs=baseA.input, outputs=baseA.layers[step].output)
        BOTA = Model(inputs=baseA.layers[step + 1].input, outputs=baseA.output)
        TOPB = Model(inputs=baseB.input, outputs=baseB.layers[step].output)
        BOTB = Model(inputs=baseB.layers[step + 1].input, outputs=baseB.output)
        for layer in TOPA.layers:
            layer.trainable = False
        for layer in TOPB.layers:
            layer.trainable = False
        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention(
                TOPA.output, TOPB.output)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(TOPA.output, TOPB.output)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention(
                TOPA.output, TOPB.output)
        elif self.args.connection_type == "Multihead":
            MLO, CC = self.MultiheadAttentionDirectSum(
                TOPA.output, TOPB.output)
        elif self.args.connection_type == "Baseline":
            MLO, CC = TOPA.output, TOPB.output
        elif self.args.connection_type == "Dot_end":
            MLO, CC = TOPA.output, TOPB.output
        MLO, CC = BOTA(MLO), BOTB(CC)
        POOLA = MaxPooling2D((7, 7))(MLO)
        POOLB = MaxPooling2D((7, 7))(CC)
        fc2 = self.Output_Block(POOLA, POOLB)
        model = Model(inputs=[TOPA.input, TOPB.input], outputs=fc2)
        return model

    def Single_View_ResNet18(self, filters=32):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        x = ZeroPadding2D((3, 3))(inputA)
        x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((7, 7), strides=(2, 2))(x)
        x = self.resnet18_start(x)
        x = self.resnet18_end(x)
        dropout = Dropout(0.3)(x)
        flattened = Flatten()(dropout)
        fc1 = Dense(224, activation="relu")(flattened)  # was 100
        fc2 = Dense(self.args.num_output_classes, activation="softmax")(fc1)
        model = Model(inputs=inputA, outputs=fc2)
        return model
    
    def mlp(self, x, hidden_units):
        for units in hidden_units:
            x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
            x = keras.layers.Dropout(self.args.dropout)(x)
        return x
    def Single_View_Vision_transformer(self):
        self.num_patches = (self.args.img_size // self.args.patch_size) ** 2
        self.transformer_units = [self.args.projection_dim * 2,self.args.projection_dim,]
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputs = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        # Augment data.
        # Create patches.
        patches = Patches(self.args.patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.args.projection_dim)(patches)
        
        # Create multiple layers of the Transformer block.
        for _ in range(self.args.transformer_layers):
            # Layer normalization 1.
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.args.num_heads, key_dim=self.args.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = keras.layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = keras.layers.Flatten()(representation)
        representation = keras.layers.Dropout(self.args.dropout)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.args.mlp_head_units)
        # Classify outputs.
        logits = keras.layers.Dense(self.args.num_output_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model
        
    def ResNet18_Segmentation(self, filters=32):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        x = ZeroPadding2D((3, 3))(inputA)
        x = Conv2D(filters, (2, 2), strides=(2, 2))(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        for ii in range(1, 3):
            x = self.identity_block(x, filters)
        x = self.con_identity_block(x, filters*2, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*2)
        x = self.con_identity_block(x, filters*4, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*4)
        x = self.con_identity_block(x, filters*8)
        # downsample here
        x = self.con_identity_block(x, filters*8)
        x = self.con_identity_block(x, filters*8)
        # downsample here
        x = self.con_identity_block(x, filters*8, stride=False)
        x = AveragePooling2D((2, 2))(x)
        model = Model(inputs=inputA, outputs=fc2)
        return model

    def Pretrained_Model_Block_Resnet18(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        #new_layer_input_name = 'conv3_block4_out'
        new_layer_input_name = 'conv2_block3_out'
        baseA = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1")
        for lay in range(0, len(baseB.layers)):
            if baseB.layers[lay].name == new_layer_input_name:
                step = lay

        TOPA = Model(inputs=baseA.input, outputs=baseA.layers[step].output)
        TOPB = Model(inputs=baseB.input, outputs=baseB.layers[step].output)
        for layer in TOPA.layers:
            layer.trainable = False
        for layer in TOPB.layers:
            layer.trainable = False

        MLO = self.Additional_resnet_layers(TOPA.output)
        CC = self.Additional_resnet_layers(TOPB.output)
        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Multihead":
            MLO, CC = self.MultiheadAttentionDirectSum(MLO, CC)
        ResBotA = self.resnet18_end(MLO)
        ResBotB = self.resnet18_end(CC)
        Out = self.Output_Block(ResBotA, ResBotB)
        model = Model(inputs=[TOPA.input, TOPB.input], outputs=Out)
        return model

    def Output_Block(self, x, y, out = None):
        combined = concatenate([x, y])
        dropout = Dropout(0.3)(combined)
        flattened = Flatten()(dropout)
        fc1 = Dense(self.args.img_size*2)(flattened)  # was 100
        norm_fc1 = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act_fc1 = Activation('relu')(norm_fc1)
        if out == None:
            fc2 = Dense(self.args.num_output_classes,activation="softmax")(act_fc1)
        else:
            fc2 = Dense(out,activation="softmax")(act_fc1)
        return fc2
    
    def Output_Block_Single(self, x, out = None):
        dropout = Dropout(0.3)(x)
        flattened = Flatten()(dropout)
        fc1 = Dense(256,activation="relu")(flattened)  # was 100
        norm_fc1 = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act_fc1 = Activation('relu')(norm_fc1)
        
        return act_fc1

    def Output_Block_Mult(self, x, y):
        x = tf.reshape(x, [-1, x.shape[1]*x.shape[2], x.shape[3]])
        y = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        reordery = tf.transpose(y, [0, 2, 1])
        
        xy = tf.matmul(x, reordery)

        flattened = Flatten()(xy)
        dropout = Dropout(0.3)(flattened)
        if out == None:
            fc2 = Dense(self.args.num_output_classes,activation="softmax")(dropout)
        else:
            fc2 = Dense(out,activation="softmax")(dropout)
        return fc2

    def Create_Segmentation_Model(self):
        if self.args.training_type == "Regression":
            print("enabling Regression")
            model = self.unet_regression()
        else:
            if self.args.num_output_objects == 1:
                if self.args.num_input_objects == 1:
                    if self.args.backbone == "U":
                        print("creating Unet")
                        model = self.unet()
                    elif self.args.backbone == "UDE":
                        model = self.unet_decon()
                    
            elif self.args.num_output_objects > 1:
                if self.args.num_input_objects > 1:
                    if self.args.backbone == "U":
                        print("creating Unet")
                        model = self.unet_dual()
                    elif self.args.backbone == "SK":
                        model = self.skunet_dual()
                    elif self.args.backbone == "U2":
                        model = self.unet_dual2()
                    elif self.args.backbone == "U3":
                        model = self.unet_dual3()
                        print("loading_split_unet")

                    

        return model

    def Create_Classification_Model(self):
        if self.args.training_type == "Regression":
            if self.args.backbone == "Resnet18":
                model = self.Resnet18_regression()
            elif self.args.backbone == "Resnet18_2":
                model = self.Resnet18_regression2()
        else:
            if self.args.pretrain == True:
                if self.args.backbone == "Resnet50":
                    model = self.Pretrained_Model_Block_Resnet50()
                elif self.args.backbone == "Resnet18":
                    print("yes")
                    model = self.Pretrained_Model_Block_Resnet18()
                elif self.args.backbone == "U2":
                    model = self.unet_dual2()
                    print("loading_split_unet")
            else:
                if self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                    if self.args.backbone == "Resnet18":
                        model = self.Single_View_ResNet18()
                    elif self.args.backbone == "EfficientNetB4":
                        model = self.Pretrained_EfficientNetB4_single()
                    elif self.args.backbone == "Densenet121":
                        model = self.Pretrained_DenseNet_single()
                    elif self.args.backbone == "Transformer":
                        model = self.Single_View_Vision_transformer()
                    elif self.args.backbone == "Resnet101":
                        model = self.Pretrained_Resnet101_single()
                        
                        
                else:
                    if self.args.backbone == "Resnet50":
                        model = self.ResNet50()
                    elif self.args.backbone == "Simple":
                        model = self.Simple_Config()
                    elif self.args.backbone == "CrossResnet50":
                        model = self.Crossview2DualAttResNet50()
                    elif self.args.backbone == "Resnet18":
                        print("Resnet18")
                        model = self.Resnet18_Dual()
                    elif self.args.backbone == "Resnet_grade":
                        model = self.Resnet18_Grade()
                    elif self.args.backbone == "Resnet_timeseries":
                        model = self.Resnet18_combined()
                    elif self.args.backbone == "Resnet_multiview":
                        model = self.Resnet18_attention_negatives()
                    elif self.args.backbone == "Resnet_multiview_subtract":
                        model = self.Resnet18_attention_subtract()
                    elif self.args.backbone == "timeseries":
                        model = self.Resnet18_time_series()
                    elif self.args.backbone == "3D_model":
                        model = self.model_3D()
                    elif self.args.backbone == "Resnet_att_timeseries":
                        model = self.Resnet18_attention()
                    elif self.args.backbone == "Resnet_combined":
                        model = self.Resnet18_mult_charateristics()
                    elif self.args.backbone == "EfficientNetB0":
                        model = self.Pretrained_EfficientNetB0()
                    elif self.args.backbone == "EfficientNetB4":
                        model = self.Pretrained_EfficientNetB4()
                    elif self.args.backbone == "Resnet_multiview_efficientnet":
                        model = self.Resnet18_attention_efficient()
                    
                

        return model
    def Pretrained_Resnet101_single(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.resnet.ResNet101(include_top=False,weights="imagenet",input_shape=(
                self.args.img_size, self.args.img_size, self.N_channels),pooling=None,classes=1000)
        baseA.trainable = False
        POOLA = AveragePooling2D((7, 7))(baseA.output)
        dropout = Dropout(self.args.dropout)(POOLA)
        flattened = Flatten()(dropout)
        fc1 = Dense(self.args.img_size*2,
                    activation="relu")(flattened)  # was 100
        fc2 = Dense(self.args.num_output_classes,activation="softmax")(fc1)
        model = Model(inputs=[baseA.input], outputs=fc2)
        return model
        
    def side_stack3(self, input_tensor, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides=(2, 2)):
        x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1))(x)
        x = BatchNormalization()(x)


        return x
    def side_stack4(self, x, nb_filter1, nb_filter2, nb_filter3, kernel_size):
        x = Conv2D(nb_filter1, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1))(x)
        x = BatchNormalization()(x)
        return x

    def identity_block5(self,input_tensorA, input_tensorB, kernel_size, filters):
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        MLO = self.side_stack4(input_tensorA, nb_filter1, nb_filter2, nb_filter3, kernel_size)
        CC = self.side_stack4(input_tensorB, nb_filter1, nb_filter2, nb_filter3, kernel_size)
        
        MLO_att = GlobalAveragePooling2D()(MLO)
        MLO_d = Dense(MLO_att.shape[1]/16, activation='relu')(MLO_att)
        MLO_d = Dense(MLO_att.shape[1], activation='sigmoid')(MLO_d)
        
        CC_att = GlobalMaxPooling2D()(CC)
        CC_d = Dense(CC_att.shape[1]/16, activation='relu')(CC_att)
        CC_d = Dense(CC_att.shape[1], activation='sigmoid')(CC_d)
        
        Combined_att = tf.keras.layers.Average()([MLO_d, CC_d])
        
        MLO_att = tf.keras.layers.Multiply()([MLO, Combined_att])
        CC_att = tf.keras.layers.Multiply()([CC, Combined_att])
        
        MLO = Add()([MLO_att, input_tensorA])
        MLO = Activation('relu')(MLO)
        CC = Add()([CC_att, input_tensorB])
        CC = Activation('relu')(CC)
        return MLO, CC
    
    def input_block2(self, x, filters=32):
        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((7, 7), strides=(2, 2))(x)
        return x
    
    def conv_block5(self, input_tensorA, input_tensorB, kernel_size, filters, strides=(2, 2)):
        
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        MLO = self.side_stack3(input_tensorA, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides = strides)
        CC = self.side_stack3(input_tensorB, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides = strides)


        MLO_att = GlobalMaxPooling2D()(MLO)
        MLO_d = Dense(MLO_att.shape[1]/16, activation='relu')(MLO_att)
        MLO_d = Dense(MLO_att.shape[1], activation='sigmoid')(MLO_d)
        
        CC_att = GlobalMaxPooling2D()(CC)
        CC_d = Dense(CC_att.shape[1]/16, activation='relu')(CC_att)
        CC_d = Dense(CC_att.shape[1], activation='sigmoid')(CC_d)
        
        Combined_att = tf.keras.layers.Average()([MLO_d, CC_d])
        
        MLO_att = tf.keras.layers.Multiply()([MLO, Combined_att])
        CC_att = tf.keras.layers.Multiply()([CC, Combined_att])
        
        
        
        MLO_shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensorA)
        MLO_shortcut = BatchNormalization(axis=-1)(MLO_shortcut)
        
        CC_shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensorB)
        CC_shortcut = BatchNormalization(axis=-1)(CC_shortcut)
        
        MLO = Add()([MLO_att, MLO_shortcut])
        MLO = Activation('relu')(MLO)
        CC = Add()([CC_att, CC_shortcut])
        CC = Activation('relu')(CC)
        
        return MLO, CC
    
    def Crossview2DualAttResNet50(self, num_grades = 2, filters=96):
        
            
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size, self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size, self.args.img_size, self.N_channels))
        
        MLO = self.input_block2(inputA)
        CC = self.input_block2(inputB)
        
        MLO, CC = self.conv_block5(MLO, CC, 3, [filters, filters, 4*filters], strides=(1, 1))
        
        
        MLO, CC = self.identity_block5(MLO, CC, 3, [filters, filters, 4*filters])
        MLO, CC = self.identity_block5(MLO, CC, 3, [filters, filters, 4*filters])
        

        MLO, CC = self.conv_block5(MLO, CC, 3, [2*filters, 2*filters, 8*filters])
        
        for i in range(0, 3):
            MLO, CC = self.identity_block5(MLO, CC, 3, [2*filters, 2*filters, 8*filters])

        MLO, CC = self.conv_block5(MLO, CC, 3, [4*filters, 4*filters, 16*filters])
        
        
        
        for i in range(0, 5):
            MLO, CC = self.identity_block5(MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        MLO, CC = self.conv_block5(MLO, CC, 3, [8*filters, 8*filters, 32*filters])
        
        MLO, CC = self.identity_block5(MLO, CC, 3, [8*filters, 8*filters, 32*filters])
        MLO, CC = self.identity_block5(MLO, CC, 3, [8*filters, 8*filters, 32*filters]) #2048
        
        MLO = MaxPooling2D((7, 7))(MLO)
        CC = MaxPooling2D((7, 7))(CC)

        combined = concatenate([MLO, CC])
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        dropout = Dropout(0.3)(combined)
        flattened = Flatten()(dropout)
        fc1 = Dense(256, activation="relu")(flattened) # was 100
        if self.args.use_output_bias:
            fc2 = Dense(2, activation="softmax", bias_initializer=self.args.output_bias)(fc1)
        else:
            fc2 = Dense(2, activation="softmax")(fc1)
        model = Model(inputs=[inputA, inputB], outputs=fc2)
        return model

    def identity_block4(self, input_tensorA, input_tensorB, kernel_size, filters):
        nb_filter1, nb_filter2, nb_filter3 = filters

        MLO = self.side_stack(input_tensorA, nb_filter1,
                              nb_filter2, nb_filter3, kernel_size)
        CC = self.side_stack(input_tensorB, nb_filter1,
                             nb_filter2, nb_filter3, kernel_size)
        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention(
                MLO, CC)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention(
                MLO, CC)
        elif self.args.connection_type == "Multihead":
            MLO, CC = self.MultiheadAttentionDirectSum(MLO, CC)
        elif self.args.connection_type == "Concat":
            MLO, CC = self.Maximum_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Baseline":
            MLO, CC = MLO, CC
        MLO = Add()([MLO, input_tensorA])
        MLO = Activation('relu')(MLO)
        CC = Add()([CC, input_tensorB])
        CC = Activation('relu')(CC)

        return MLO, CC
    def Pretrained_DenseNet_single(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.DenseNet121(include_top=False,weights="imagenet",input_shape=(
                self.args.img_size, self.args.img_size, self.N_channels),pooling=None,classes=1000)
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        dropout = Dropout(self.args.dropout)(POOLA)
        flattened = Flatten()(dropout)
        fc1 = Dense(self.args.img_size*2,
                    activation="relu")(flattened)  # was 100
        fc2 = Dense(self.args.num_output_classes,activation="softmax")(fc1)
        model = Model(inputs=[baseA.input], outputs=fc2)
        return model

    def conv_block4(self, input_tensorA, input_tensorB, kernel_size, filters, strides=(2, 2)):

        nb_filter1, nb_filter2, nb_filter3 = filters

        MLO = self.side_stack2(
            input_tensorA, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides=strides)
        CC = self.side_stack2(input_tensorB, nb_filter1,
                              nb_filter2, nb_filter3, kernel_size, strides=strides)
        
        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention(
                MLO, CC)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention(
                MLO, CC)
        elif self.args.connection_type == "Multihead":
            MLO, CC = self.MultiheadAttentionDirectSum(MLO, CC)
        elif self.args.connection_type == "Concat":
            MLO, CC = self.Maximum_Excitation_Attention(MLO, CC)
        elif self.args.connection_type == "Baseline":
            MLO, CC = MLO, CC
        
        MLO_shortcut = Conv2D(nb_filter2, (3, 3),
                              strides=strides)(input_tensorA)
        MLO_shortcut = BatchNormalization(
            momentum=0.1, epsilon=0.000001)(MLO_shortcut)

        CC_shortcut = Conv2D(nb_filter2, (3, 3),
                             strides=strides)(input_tensorB)
        CC_shortcut = BatchNormalization(
            momentum=0.1, epsilon=0.000001)(CC_shortcut)
        
        MLO = Add()([MLO, MLO_shortcut])
        MLO = Activation('relu')(MLO)

        CC = Add()([CC, CC_shortcut])
        CC = Activation('relu')(CC)
        return MLO, CC

    def side_stack(self, x, nb_filter1, nb_filter2, nb_filter3, kernel_size):
        x = Conv2D(nb_filter1, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        #x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same')(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)

        #x = Conv2D(nb_filter3, (1, 1))(x)
        #x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        #x = Activation('relu')(x)
        return x

    def side_stack2(self, input_tensor, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides=(2, 2)):
        x = Conv2D(nb_filter1, (3, 3), strides=strides)(input_tensor)
        x = BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
        x = Activation('relu')(x)
        #x = ZeroPadding2D((3, 3))(x)

        x = Conv2D(nb_filter2, (3,3), padding='same')(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
        x = Activation('relu')(x)

        #x = Conv2D(nb_filter3, (1, 1))(x)
        #x = BatchNormalization(momentum=0.1, epsilon=0.00001)(x)
        #x = Activation('relu')(x)
        
        return x

    def MultiheadAttentionDirectSum(self, x, y):
        x, y = self.crossview_component(x, y)
        return x, y

    def DualResNet50_TOP(self, MLO, CC, filters=64):
        MLO, CC = self.conv_block4(
            MLO, CC, 3, [filters, filters, 4*filters], strides=(1, 1))
        for i in range(0, 1):
            MLO, CC = self.identity_block4(
                MLO, CC, 3, [filters, filters, 4*filters])
        
        MLO, CC = self.conv_block4(
            MLO, CC, 3, [2*filters, 2*filters, 8*filters])

        for i in range(0, 1):
            MLO, CC = self.identity_block4(
                MLO, CC, 3, [2*filters, 2*filters, 8*filters])

        return MLO, CC

    def DualSimple_TOP(self, MLO, CC, filters=64):
        MLO, CC = self.conv_block4(
            MLO, CC, 3, [filters, filters, 4*filters], strides=(1, 1))

        MLO, CC = self.identity_block4(
            MLO, CC, 3, [filters, filters, 4*filters])
        return MLO, CC

    def augment(self, x):
        #x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        #x = tf.keras.layers.RandomRotation(0.2)(x)
        return x

    def Input_Block(self, inputA, inputB):

        #AugA = self.augment(inputA)
        #AugB = self.augment(inputB)
        MLO = self.input_block(inputA)
        CC = self.input_block(inputB)
        return MLO, CC
    
    def Input_Block_Single(self, inputA):
        #AugA = self.augment(inputA)
        #AugB = self.augment(inputB)
        #MLO = LayerNormalization(epsilon=1e-6)(inputA)
        MLO = self.input_block(inputA)
        return MLO
    
    def input_block_3D(self, x, filters=32):
        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = tf.reshape(x,[-1, x.shape[1], x.shape[2], 1, x.shape[3]])
        return x
    
    def input_block(self, x, filters=64):
        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def DualResNet50_BOT(self, MLO, CC, filters=64):

        MLO, CC = self.conv_block4(
            MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        for i in range(0, 1):
            MLO, CC = self.identity_block4(
                MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        MLO, CC = self.conv_block4(
            MLO, CC, 3, [8*filters, 8*filters, 32*filters])

        for i in range(0, 1):
            MLO, CC = self.identity_block4(
                MLO, CC, 3, [8*filters, 8*filters, 32*filters])

        MLO = AveragePooling2D((2, 2))(MLO)
        CC = AveragePooling2D((2, 2))(CC)
        return MLO, CC

    def DualSimple_BOT(self, MLO, CC, filters=64):

        MLO, CC = self.conv_block4(
            MLO, CC, 3, [4*filters, 4*filters, 16*filters])
        MLO, CC = self.identity_block4(
            MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        MLO = MaxPooling2D((7, 7))(MLO)
        CC = MaxPooling2D((7, 7))(CC)
        return MLO, CC

    def DualResNet50_BOT_nopool(self, MLO, CC, filters=64):

        MLO, CC = self.conv_block4(
            MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        for i in range(0, 5):
            MLO, CC = self.identity_block4(
                MLO, CC, 3, [4*filters, 4*filters, 16*filters])

        MLO, CC = self.conv_block4(
            MLO, CC, 3, [8*filters, 8*filters, 32*filters])

        MLO, CC = self.identity_block4(
            MLO, CC, 3, [8*filters, 8*filters, 32*filters])
        MLO, CC = self.identity_block4(
            MLO, CC, 3, [8*filters, 8*filters, 32*filters])  # 2048

        return MLO, CC

    def ResNet50(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA, ResTopB = self.DualResNet50_TOP(InputA, InputB)
        
        if self.args.connection_type != "Dot_end":
            ResBotA, ResBotB = self.DualResNet50_BOT(ResTopA, ResTopB)
            Out = self.Output_Block(ResBotA, ResBotB)
        else:
            ResBotA, ResBotB = self.DualResNet50_BOT_nopool(ResTopA, ResTopB)
            Out = self.Output_Block_Mult(ResBotA, ResBotB)
        
        model = Model(inputs=[inputA, inputB], outputs=Out)
        
        return model

    def Simple_Config(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA, ResTopB = self.DualSimple_TOP(InputA, InputB)
        if self.args.connection_type == "Average":
            CrossA, CrossB = self.Averaged_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Self":
            CrossA, CrossB = self.Self_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Add":
            CrossA, CrossB = self.Addition_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Multihead":
            CrossA, CrossB = self.MultiheadAttentionDirectSum(ResTopA, ResTopB)
        elif self.args.connection_type == "Baseline":
            CrossA, CrossB = ResTopA, ResTopB
        elif self.args.connection_type == "Dot_end":
            CrossA, CrossB = ResTopA, ResTopB

        if self.args.connection_type != "Dot_end":
            ResBotA, ResBotB = self.DualSimple_BOT(CrossA, CrossB)
            Out = self.Output_Block(ResBotA, ResBotB)
        else:
            ResBotA, ResBotB = self.DualResNet50_BOT_nopool(CrossA, CrossB)
            Out = self.Output_Block_Mult(ResBotA, ResBotB)
        model = Model(inputs=[inputA, inputB], outputs=Out)
        return model

    def Create_Segmentation22_Model(self):
        base = keras.applications.EfficientNetB7(input_shape=[
                                                 self.image_size, self.image_size, self.N_channels], include_top=False, weights=None)

        base.load_weights("WIEGHTS/256_eff_base.h5")

        skip_names = ['block1a_activation',  # size 64*64
                      'block2g_activation',  # size 32*32
                      'block3g_activation',  # size 16*16
                      'block5g_activation',  'top_activation']

        skip_outputs = [base.get_layer(name).output for name in skip_names]

        downstack = keras.Model(inputs=base.input, outputs=skip_outputs)
        downstack.trainable = False

        # Four upstack blocks for upsampling sizes
        # 4->8, 8->16, 16->32, 32->64
        upstack = [pix2pix.upsample(512, self.N_channels),
                   pix2pix.upsample(256, self.N_channels),
                   pix2pix.upsample(128, self.N_channels),
                   pix2pix.upsample(64, self.N_channels)]

        # We can explore the individual layers in each upstack block.
        # upstack[0].layers

        inputs = keras.layers.Input(
            shape=[self.image_size, self.image_size, self.N_channels])

        # downsample
        down = downstack(inputs)
        out = down[-1]
        # prepare skip-connections
        skips = reversed(down[:-1])
        # choose the last layer at first 4 --> 8
        # upsample with skip-connections
        for up, skip in zip(upstack, skips):
            out = up(out)
            out = keras.layers.Concatenate()([out, skip])
        # define the final transpose conv layer
        # image 128 by 128 with 59 classes
        out = keras.layers.Conv2DTranspose(
            self.No_output + 1, self.N_channels, strides=2, padding='same')(out)
        # complete unet model
        unet = keras.Model(inputs=inputs, outputs=out)
        return unet

    def tokenizer(self, x, tokens):
        flatten_A = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])
        embedA = tf.keras.layers.Conv1D(tokens, kernel_size=1)(flatten_A)
        soft = tf.nn.softmax(embedA, axis=1)
        matrixmult_overbatch = tf.matmul(flatten_A, soft)
        embed2 = tf.keras.layers.Conv1D(
            tokens, kernel_size=1)(matrixmult_overbatch)
        reorder = tf.transpose(embed2, [0, 2, 1])
        map_token = Dense(reorder.shape[2], activation=None)(reorder)
        reorder2 = tf.transpose(map_token, [0, 2, 1])
        reorder_flattened = tf.transpose(flatten_A, [0, 2, 1])

        matrixmult_overbatch2 = tf.matmul(map_token, reorder_flattened)
        reorder_matrixmult_overbatch2 = tf.transpose(
            matrixmult_overbatch2, [0, 2, 1])
        soft2 = tf.nn.softmax(reorder_matrixmult_overbatch2, axis=1)
        matrixmult_overbatch3 = tf.matmul(reorder_flattened, soft2)

        return matrixmult_overbatch3, soft2

    def embed_tensor(self, x, embedding=32, heads=12):

        embed2 = tf.keras.layers.Conv1D(embedding*heads, kernel_size=1)(x)

        return embed2

    def reverse_tokenizer(self, y, att):
        tran_att = tf.transpose(att, [0, 2, 1])
        matrixmult_overbatch = tf.matmul(y, tran_att)

        return matrixmult_overbatch

    def scaled_dot_product(self, q, k, v, i):
        # calculates Q . K(transpose)
        Q = q[:, i, :, :]
        K = k[:, i, :, :]
        reorderQ = tf.transpose(Q, [0, 2, 1])
        qkt = tf.matmul(reorderQ, K)
        # caculates scaling factor
        dk = tf.math.sqrt(tf.cast(reorderQ.shape[-1], dtype=tf.float32))
        scaled_qkt = qkt/dk
        softmax = tf.nn.softmax(scaled_qkt, axis=-1)

        z = tf.matmul(softmax, v)
        #z = tf.reshape(z, [-1, 1, z.shape[1], z.shape[2]])
        # shape: (m,Tx,depth), same shape as q,k,v
        return z

    def in_add_linear(self, x, y, drop):
        conx = tf.keras.layers.Conv1D(x.shape[-1], kernel_size=1)(x)
        dropoutx = Dropout(drop)(conx)
        addition = dropoutx + y
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(addition)
        return norm

    def multiheadatt(self, Q, K, V, heads, emb):
        multi_attn = []
        for i in range(heads):
            multi_attn.append(self.scaled_dot_product(Q, K, V, i))
        multi_head = tf.concat(multi_attn, axis=1)
        reorder_mult_head = tf.transpose(multi_head, [0, 2, 1])
        #multi_head_attention = Dense(V.shape[1])(reorder_mult_head)
        multi_head_attention = tf.keras.layers.Conv1D(
            V.shape[1], kernel_size=1)(reorder_mult_head)

        return multi_head_attention

    def Resnet18(self, out = None):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA = self.resnet18_start(InputA)
        ResTopB = self.resnet18_start(InputB)
        if self.args.connection_type == "Average":
            CrossA, CrossB = self.Averaged_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Self":
            CrossA, CrossB = self.Self_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Add":
            CrossA, CrossB = self.Addition_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Multihead":
            CrossA, CrossB = self.MultiheadAttentionDirectSum(ResTopA, ResTopB)
        elif self.args.connection_type == "Baseline":
            CrossA, CrossB = ResTopA, ResTopB

        ResBotA = self.resnet18_end(CrossA)
        ResBotB = self.resnet18_end(CrossB)
        if out == None:
            Out = self.Output_Block(ResBotA, ResBotB)
        else:
            Out = self.Output_Block(ResBotA, ResBotB, out=3)
        model = Model(inputs=[inputA, inputB], outputs=Out)

        return inputA, inputB, model, Out
    
    def Resnet18_Single_Features(self, out = None):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA = self.Input_Block_Single(inputA)
        ResTopA = self.resnet18_start(InputA)
        ResBotA = self.resnet18_end(ResTopA)
        Out = self.Output_Block_Single(ResBotA)
        model = Model(inputs=[inputA], outputs=ResBotA)

        return inputA, model, Out
    
    def Resnet18_Single_End(self, x, out = None):
        InputA = self.Input_Block_Single(x)
        ResTopA = self.resnet18_start(InputA)
        ResBotA = self.resnet18_end(ResTopA)
        Out = self.Output_Block_Single(ResBotA)


        return Out
    
    def Resnet18_Output_Block(self, out = None):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA = self.Input_Block_Single(inputA)
        ResTopA = self.resnet18_start(InputA)
        ResBotA = self.resnet18_end(ResTopA)
        Out = self.Output_Block_Single(ResBotA)
        model = Model(inputs=[inputA], outputs=Out)

        return inputA, model, Out
    
    def Resnet18_Dual(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA = self.resnet18_start(InputA)
        ResTopB = self.resnet18_start(InputB)
        if self.args.connection_type == "Average":
            CrossA, CrossB = self.Averaged_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Self":
            CrossA, CrossB = self.Self_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Add":
            CrossA, CrossB = self.Addition_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Multihead":
            CrossA, CrossB = self.MultiheadAttentionDirectSum(ResTopA, ResTopB)
        elif self.args.connection_type == "Baseline":
            CrossA, CrossB = ResTopA, ResTopB

        ResBotA = self.resnet18_end(CrossA)
        ResBotB = self.resnet18_end(CrossB)
        Out = self.Output_Block(ResBotA, ResBotB)
        model = Model(inputs=[inputA, inputB], outputs=Out)

        return model

    def Resnet18_combined(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA, inputB, model_top1, out1 = self.Resnet18()
        inputC, inputD, model_top2, out2 = self.Resnet18()
        #
        model_top1.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Normal_Maglignant_Class.h5")
        model_top2.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Normal_Maglignant_Class.h5")
        #model_top1.load_weights("WIEGHTS/Baseline_0_1_512_40_Resnet18_Normal_Malignant.h5")
        #model_top2.load_weights("WIEGHTS/Baseline_0_1_512_40_Resnet18_Normal_Malignant.h5")
        model_normal = Model(inputs=[inputA, inputB], outputs=model_top1.layers[-2].output)
        model_normal.trainable = False

        model_pre = Model(inputs=[inputC, inputD],outputs=model_top2.layers[-2].output)
        model_pre.trainable = False
        diff = keras.layers.Subtract()([model_normal.output,model_pre.output])
        inputE = Input(shape=(1))
        #encodeE = keras.layers.Dense(diff.shape[1], activation="linear")(inputE)
        #scale_layer = tf.multiply(diff, inputE)
        scale_layer = tf.divide(diff, inputE)
        
        #Reshape = tf.reshape(scale_layer,[-1, 1, scale_layer.shape[1]])
        #Normal_out = tf.reshape(model_normal.output,[-1, model_normal.output.shape[1],1])
        #Pre_out = tf.reshape(model_pre.output, [-1, model_pre.output.shape[1],1])

        #diff = concatenate([model_normal.output, model_pre.output], axis=-2)
        #diff = concatenate([Normal_out,Pre_out], axis=-1)
        #conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(Reshape)
        #conv1 = keras.layers.BatchNormalization()(conv1)
        #conv1 = keras.layers.ReLU()(conv1)

        #conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        #conv2 = keras.layers.BatchNormalization()(conv2)
        #conv2 = keras.layers.ReLU()(conv2)

        #conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        #conv3 = keras.layers.BatchNormalization()(conv3)
        #conv3 = keras.layers.ReLU()(conv3)

        #gap = keras.layers.GlobalAveragePooling1D()(conv3)
        step = keras.layers.Dense(scale_layer.shape[1], activation="relu")(scale_layer)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(step)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD], inputE], outputs=output_layer)
        #model.summary()
        return model
    
    def Resnet18_time_series(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA, inputB, model_top1, out1 = self.Resnet18()
        inputC, inputD, model_top2, out2 = self.Resnet18()
        #
        model_top1.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Normal_Maglignant_Class.h5")
        model_top2.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Normal_Maglignant_Class.h5")
        model_normal = Model(inputs=[inputA, inputB], outputs=model_top1.layers[-2].output)
        model_normal.trainable = False

        model_pre = Model(inputs=[inputC, inputD],outputs=model_top2.layers[-2].output)
        model_pre.trainable = False
        #diff = keras.layers.Subtract()([model_normal.output,model_pre.output])
        #inputE = Input(shape=(1))
        #encodeE = keras.layers.Dense(diff.shape[1], activation="linear")(inputE)
        #scale_layer = tf.multiply(diff, inputE)
        #scale_layer = tf.divide(diff, encodeE)
        
        #Reshape = tf.reshape(scale_layer,[-1, 1, scale_layer.shape[1]])
        Normal_out = tf.reshape(model_normal.output,[-1, model_normal.output.shape[1],1])
        Pre_out = tf.reshape(model_pre.output, [-1, model_pre.output.shape[1],1])

        #diff = concatenate([model_normal.output, model_pre.output], axis=-2)
        diff = concatenate([Normal_out,Pre_out], axis=-1)
        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(diff)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)
        #step = keras.layers.Dense(scale_layer.shape[1], activation="relu")(gap)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(gap)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD]], outputs=output_layer)
        #model.summary()
        return model
    
    def Resnet18_mult_charateristics(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA, inputB, model_top1, out1 = self.Resnet18()
        inputC, inputD, model_top2, out2 = self.Resnet18()
        inputE, inputF, model_top3, out3 = self.Resnet18()
        
        
        model_top1.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Mass.h5")
        model_top2.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Invasivetype.h5")
        model_top3.load_weights("WIEGHTS/Baseline_0_0_512_40_Resnet18_Calcifications.h5")
        model_mass = Model(inputs=[inputA, inputB], outputs=model_top1.layers[-2].output)
        model_mass.trainable = False

        model_inv = Model(inputs=[inputC, inputD],outputs=model_top2.layers[-2].output)
        model_inv.trainable = False
        
        model_cal = Model(inputs=[inputE, inputF],outputs=model_top3.layers[-2].output)
        model_cal.trainable = False
        
        #diff = concatenate([model_mass.output, model_inv.output, model_cal.output], axis=-1)

        combine_mass = concatenate([model_mass.output, model_inv.output], axis=-1)
        combined_dense = Dense(model_mass.output.shape[1], activation="relu")(combine_mass)
        combine_cal = concatenate([combined_dense, model_cal.output], axis=-1)
        
        #gap = keras.layers.GlobalAveragePooling1D()(conv3)
        step = keras.layers.Dense(model_mass.output.shape[1], activation="relu")(combine_cal)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(step)
        model = Model(inputs=[[inputA, inputB],[inputC, inputD],[inputE, inputF]], outputs=output_layer)
        #model.summary()
        return model

    def Resnet18_attention(self, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA, inputB, model_top1, out1 = self.Resnet18()
        inputC, inputD, model_top2, out2 = self.Resnet18()
        #
        #model_top1.load_weights("WIEGHTS/Baseline_1_0_0_512_60_Resnet18_Grade_test.h5")
        #model_top2.load_weights("WIEGHTS/Baseline_1_0_0_512_60_Resnet18_Grade_test.h5")
        model_normal = Model(
            inputs=[inputA, inputB], outputs=model_top1.layers[-2].output)
        #model_normal.trainable = False

        model_pre = Model(inputs=[inputC, inputD],
                          outputs=model_top2.layers[-2].output)
        #model_pre.trainable = False
        

        #diff = keras.layers.Subtract()([left, right])
        #inputE = Input(shape=(1))
        #encode_time = Dense(diff.shape[1], activation="linear")(inputE)
        #scale_layer = tf.multiply(diff, encode_time)
        #scale_layer = tf.divide(diff, encode_time)
        #Reshape = tf.reshape(scale_layer,[-1, scale_layer.shape[1], 1])
        Normal_out = tf.reshape(model_normal.output,[-1, model_normal.output.shape[1], 1])
        Pre_out = tf.reshape(model_pre.output, [-1, model_pre.output.shape[1], 1])
        left = Dense(Normal_out.shape[1], activation="relu")(Normal_out)
        right = Dense(Pre_out.shape[1], activation="relu")(Pre_out)
        #diff = concatenate([model_normal.output,model_pre.output])
        #diff = concatenate([Normal_out, Pre_out], axis=-1)
        #diff = keras.layers.Subtract()([model_normal.output,model_pre.output])
        #x = Reshape
        #for _ in range(num_transformer_blocks):
            #x = self.transformer_encoder(x)
        #x = GlobalAveragePooling1D(data_format="channels_first")(x)
        #for dim in mlp_units:
            #x = Dense(dim, activation="relu")(x)
            #x = Dropout(mlp_dropout)(x)
        #matrix_multiplication = tf.matmul(left, right, transpose_b=True)
        diff = keras.layers.Subtract()([left, right])
        #POOL = AveragePooling2D(pool_size=(1, 1), padding='same')(matrix_multiplication)
        POOL = GlobalAveragePooling1D()(diff)
        Drop = Dropout(dropout)(POOL)
        #flattened = Flatten()(Drop)
        fc1 = keras.layers.Dense(diff.shape[1], activation="relu")(Drop)
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act1 = Activation("relu")(norm)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(act1)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD]], outputs=output_layer)
        return model
    
    def subtract_squared_layer(self, r1, r2):
        minus_r2 = Lambda(lambda x: -x)(r2)
        subtracted = add([r1,minus_r2])
        out= Lambda(lambda x: x**2)(subtracted)
        return out
    
    def Resnet18_attention_subtract(self, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        
        inputA, model_topA, outA = self.Resnet18_Single_Features()
        inputB, model_topB, outB = self.Resnet18_Single_Features()
        inputC, model_topC, outC = self.Resnet18_Single_Features()
        inputD, model_topD, outD = self.Resnet18_Single_Features()
        
        #model_pre.trainable = False
        
        left_diff = keras.layers.Subtract()([model_topA.output, model_topC.output])
        right_diff = keras.layers.Subtract()([model_topB.output, model_topD.output])
        
        #left_diff = self.subtract_squared_layer(model_topA.output, model_topC.output)
        #right_diff = self.subtract_squared_layer(model_topB.output, model_topD.output)
        
        left_diff_reshape = tf.reshape(left_diff,[-1, left_diff.shape[1], 1])
        right_diff_reshape = tf.reshape(right_diff,[-1, right_diff.shape[1], 1])
        
        
        centre_mult = tf.matmul(left_diff_reshape, right_diff_reshape, transpose_b=True)
        centre_drop = Dropout(dropout)(centre_mult)
        centre_reshape = tf.reshape(centre_drop,[-1, centre_drop.shape[1],centre_drop.shape[2] , 1])
        x = Conv2D(256, (3, 3), strides=(2, 2))(centre_reshape)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        
        fc1 = keras.layers.Dense(x.shape[-1], activation="relu")(x)
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act1 = Activation("relu")(norm)
        #outbottom = self.Resnet18_Single_End(centre_reshape)
        
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(act1)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD]], outputs=output_layer)
        return model
    
    def Resnet18_regression(self, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        
        inputA, model_topA, outA = self.Resnet18_Single_Features()
        #inputB, model_topB, outB = self.Resnet18_Single_Features()
        
        #POOLA = MaxPooling2D((3, 3))(baseA.output)
        #baseA.trainable = False
        
        #left_diff = keras.layers.Subtract()([model_topA.output, model_topB.output])
        x = Conv2D(256, (1, 1))(model_topA.output)
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        act1 = Activation("relu")(norm)
        x = GlobalMaxPooling2D()(act1)
        x = Dropout(0.25)(x)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="sigmoid")(x)
        model = Model(inputs=inputA, outputs=output_layer)
        return model
    
    def Resnet18_regression2(self, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        
        inputA, model_topA, outA = self.Resnet18_Single_Features()
        inputB, model_topB, outB = self.Resnet18_Single_Features()
        
        
        #model_pre.trainable = False
        
        left_diff = keras.layers.Subtract()([model_topA.output, model_topB.output])
        #FA = Flatten()(left_diff)
        fc1 = keras.layers.Dense(left_diff.shape[-1], activation="relu")(left_diff)
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act1 = Activation("relu")(norm)

        
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="sigmoid")(act1)
        model = Model(inputs=[inputA, inputB], outputs=output_layer)
        return model
    
    def model_3D(self, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        
        inputA = Input(shape=(self.args.img_size,self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,self.args.img_size, self.N_channels))
        inputC = Input(shape=(self.args.img_size,self.args.img_size, self.N_channels))
        inputD = Input(shape=(self.args.img_size,self.args.img_size, self.N_channels))
        
        A = self.input_block_3D(inputA, filters=32)
        B = self.input_block_3D(inputB, filters=32)
        C = self.input_block_3D(inputC, filters=32)
        D = self.input_block_3D(inputD, filters=32)
        
        print(A.shape)
        left_combined = []
        for ii in range(0, A.shape[-1]):
            combine = tf.matmul(A[:,:,:,ii], B[:,:,:,ii], transpose_b=True)
            left_reshape = tf.reshape(combine,[-1, combine.shape[1], combine.shape[2], 1])
            left_combined.append(left_reshape)
        left_concat = tf.concat(left_combined, -1)
        print(left_concat.shape)

            
        
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(act1)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD]], outputs=output_layer)
        return model
    
    def Resnet18_attention_efficient(self, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        baseA = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseC = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseD = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(
            self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseB.layers:
            layer._name = layer.name + str("_1_1")
        for layer in baseB.weights:
            layer._name = layer.name + str("_1_1")
        for layer in baseC.layers:
            layer._name = layer.name + str("_1_2")
        for layer in baseC.weights:
            layer._name = layer.name + str("_1_2")
        for layer in baseD.layers:
            layer._name = layer.name + str("_1_3")
        for layer in baseD.weights:
            layer._name = layer.name + str("_1_3")
        POOLA = MaxPooling2D((7, 7), strides=(2, 2))(baseA.output)
        POOLB = MaxPooling2D((7, 7), strides=(2, 2))(baseB.output)
        POOLC = MaxPooling2D((7, 7), strides=(2, 2))(baseC.output)
        POOLD = MaxPooling2D((7, 7), strides=(2, 2))(baseD.output)
        
        #model_pre.trainable = False
        
        left_diff = keras.layers.Subtract()([POOLA, POOLC])
        right_diff = keras.layers.Subtract()([POOLB, POOLD])
        FA = Flatten()(left_diff)
        FB = Flatten()(right_diff)
        #left_diff = self.subtract_squared_layer(DA, DC)
        #right_diff = self.subtract_squared_layer(DB, DD)
        DA = Dense(128, activation="relu")(FA)
        DB = Dense(128, activation="relu")(FB)
        left_diff_reshape = tf.reshape(DA,[-1, DA.shape[1], 1])
        right_diff_reshape = tf.reshape(DB,[-1, DB.shape[1], 1])
        
        
        centre_mult = tf.matmul(left_diff_reshape, right_diff_reshape, transpose_b=True)
        centre_drop = Dropout(dropout)(centre_mult)
        centre_reshape = tf.reshape(centre_drop,[-1, centre_drop.shape[1],centre_drop.shape[2] , 1])
        x = Conv2D(256, (7, 7), strides=(2, 2))(centre_reshape)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = GlobalMaxPooling2D()(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        
        fc1 = keras.layers.Dense(256, activation="relu")(x)
        norm = BatchNormalization(momentum=0.1, epsilon=0.0001)(fc1)
        act1 = Activation("relu")(norm)
        #outbottom = self.Resnet18_Single_End(centre_reshape)
        
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(act1)
        model = Model(inputs=[[baseA.input, baseB.input], [baseC.input, baseD.input]], outputs=output_layer)
        return model
    
    def Resnet18_attention_negatives(self, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        
        inputA, model_topA, outA = self.Resnet18_Single_Features()
        inputB, model_topB, outB = self.Resnet18_Single_Features()
        inputC, model_topC, outC = self.Resnet18_Single_Features()
        inputD, model_topD, outD = self.Resnet18_Single_Features()
        
        #model_pre.trainable = False
        

        
        model_topA_out = tf.reshape(model_topA.output,[-1, model_topA.output.shape[1], 1])
        model_topB_out = tf.reshape(model_topB.output,[-1, model_topB.output.shape[1], 1])
        model_topC_out = tf.reshape(model_topC.output,[-1, model_topC.output.shape[1], 1])
        model_topD_out = tf.reshape(model_topD.output,[-1, model_topD.output.shape[1], 1])
        
        left = tf.matmul(model_topA_out, model_topC_out, transpose_b=True)
        right = tf.matmul(model_topB_out, model_topD_out, transpose_b=True)
        left_flat = Flatten()(left)
        right_flat = Flatten()(right)
        left_dense = Dense(80, activation="relu")(left_flat)
        norm_left = LayerNormalization(epsilon=1e-6)(left_dense)
        act_left = Activation("relu")(norm_left)
        right_dense = Dense(80, activation="relu")(right_flat)
        norm_right = LayerNormalization(epsilon=1e-6)(right_dense)
        act_right = Activation("relu")(norm_right)
        re_left = tf.reshape(act_left,[-1, act_left.shape[1], 1])
        re_right = tf.reshape(act_right,[-1, act_right.shape[1], 1])
        centre_out = tf.matmul(re_left, re_right, transpose_b=True)
        re_centre = tf.reshape(centre_out,[-1, centre_out.shape[1], centre_out.shape[2], 1])
        x = Conv2D(64, (7, 7), strides=(2, 2))(re_centre)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        x = AveragePooling2D((4, 4), strides=(2, 2))(x)
        #x = Conv2D(128, (3, 3))(x)
        #x = LayerNormalization(epsilon=1e-6)(x)
        #x = Activation('relu')(x)
        #x = AveragePooling2D((3, 3), strides=(2, 2))(x)
        #x = Dropout(dropout)(x)
        flattened = Flatten()(x)
        fc1 = keras.layers.Dense(self.args.img_size, activation="relu")(flattened)
        norm = LayerNormalization(epsilon=1e-6)(fc1)
        act1 = Activation("relu")(norm)
        output_layer = keras.layers.Dense(self.args.num_output_classes, activation="softmax")(act1)
        model = Model(inputs=[[inputA, inputB], [inputC, inputD]], outputs=output_layer)
        return model
    
    

    def transformer_encoder(self, inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.25):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = Add()([x, inputs])

        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return Add()([x, res])

    def Resnet18_Grade(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA = self.resnet18_start(InputA)
        ResTopB = self.resnet18_start(InputB)
        if self.args.connection_type == "Average":
            CrossA, CrossB = self.Averaged_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Self":
            CrossA, CrossB = self.Self_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Add":
            CrossA, CrossB = self.Addition_Excitation_Attention(
                ResTopA, ResTopB)
        elif self.args.connection_type == "Multihead":
            CrossA, CrossB = self.MultiheadAttentionDirectSum(ResTopA, ResTopB)
        elif self.args.connection_type == "Baseline":
            CrossA, CrossB = ResTopA, ResTopB

        ResBotA = self.resnet18_end(CrossA)
        ResBotB = self.resnet18_end(CrossB)
        dropoutA = Dropout(0.3)(ResBotA)
        flattenedA = Flatten()(dropoutA)
        A1 = Dense(self.args.img_size, activation="gelu")(
            flattenedA)  # was 100
        dropoutB = Dropout(0.3)(ResBotB)
        flattenedB = Flatten()(dropoutB)
        B1 = Dense(self.args.img_size, activation="gelu")(
            flattenedB)  # was 100
        Difference = keras.layers.Subtract()([A1, B1])
        Out = Dense(self.args.num_output_classes,
                    activation="softmax")(Difference)
        model = Model(inputs=[inputA, inputB], outputs=Out)
        return model
    

    def inital_block(self, x, filters):
        x = Conv2D(filters, (7, 7), strides=(2, 2), padding="same")(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation("gelu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x

    def identity_block(self, z, filters):
        x = Conv2D(filters, (3, 3), padding="same")(z)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        #x = LayerNormalization(epsilon=1e-6)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        #x = LayerNormalization(epsilon=1e-6)(x)
        x = Add()([x, z])
        return x

    def resnet18_start(self, x, filters=64):
        x = self.inital_block(x, filters)
        for ii in range(1, 3):
            x = self.identity_block(x, filters)

        x = self.con_identity_block(x, filters*2, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*2)
        x = self.con_identity_block(x, filters*4, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*4)
        return x

    def Additional_resnet_layers(self, x, filters=64):
        x = self.con_identity_block(x, filters*2, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*2)
        x = self.con_identity_block(x, filters*4, stride=True)
        # downsample here
        x = self.con_identity_block(x, filters*4)
        return x

    def resnet18_end(self, x, filters=64):
        x = self.con_identity_block(x, filters*8)
        # downsample here
        x = self.con_identity_block(x, filters*8)
        x = self.con_identity_block(x, filters*8)
        # downsample here
        x = self.con_identity_block(x, filters*8, stride=False)
        #x = AveragePooling2D((4,4))(x)
        #x = MaxPooling2D((4, 4))(x)
        return x

    def con_identity_block(self, x, filters, stride=False):
        if stride == True:
            x = Conv2D(filters, (3, 3), strides=(2, 2), padding="same")(x)
        else:
            x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization(momentum=0.1, epsilon=0.0001)(x)
        return x

    def crossview_component(self, x, y, tokenizer_a=False, tokenizer_b=False, num_heads=12, embedding=32, dropout=0.1):
        tokensA = x
        tokensB = y
        if tokenizer_a == True:
            tokensA, attA = self.tokenizer(x, tokens)
            reorder2 = tf.transpose(tokensA, [0, 2, 1])
            embed_A = self.embed_tensor(reorder2, heads=num_heads)

        else:
            flatten_A = tf.reshape(
                tokensA, [-1, tokensA.shape[1] * tokensA.shape[2], tokensA.shape[3]])
            embed_A = self.embed_tensor(flatten_A, heads=num_heads)

        if tokenizer_b == True:
            tokensB, attB = self.tokenizer(y, tokens)
            reorderB = tf.transpose(tokensB, [0, 2, 1])
            embed_B = self.embed_tensor(reorderB, heads=num_heads)
        else:
            flatten_B = tf.reshape(
                tokensB, [-1, tokensB.shape[1] * tokensB.shape[2], tokensB.shape[3]])
            embed_B = self.embed_tensor(flatten_B, heads=num_heads)

        reshape_embeddedQ = tf.reshape(
            embed_A, [-1, embed_A.shape[1], num_heads, embedding])
        reorderQ = tf.transpose(reshape_embeddedQ, [0, 2, 3, 1])
        reshape_embeddedK = tf.reshape(
            embed_B, [-1, embed_B.shape[1], num_heads, embedding])
        reorderK = tf.transpose(reshape_embeddedK, [0, 2, 3, 1])
        flattenVB = tf.reshape(y, [-1, y.shape[1] * y.shape[2], y.shape[3]])
        flattenVA = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])
        #reorderV = tf.transpose(FlattenV, [0, 2, 1])
        multA = self.multiheadatt(
            reorderQ, reorderK, flattenVB, num_heads, embedding)
        multB = self.multiheadatt(
            reorderK, reorderQ, flattenVA, num_heads, embedding)
        reordermultA = tf.transpose(multA, [0, 2, 1])
        reshapemultA = tf.reshape(
            reordermultA, [-1, x.shape[1], x.shape[2], x.shape[3]])
        linearA_addition = self.in_add_linear(reshapemultA, y, dropout)

        reordermultB = tf.transpose(multB, [0, 2, 1])
        reshapemultB = tf.reshape(
            reordermultB, [-1, y.shape[1], y.shape[2], y.shape[3]])
        linearB_addition = self.in_add_linear(reshapemultB, x, dropout)

        #reshape_V = tf.reshape(tokensB, [-1,num_heads, tokensB.shape[-1], tokensA.shape[-1]])
        #multi = MultiHeadAttention(d_model=reshape_embeddedA.shape[-1], num_heads=num_heads)
        #out, attn = multi(reshape_embeddedA, k=reshape_embeddedB, q=embed_B, mask=None)

        #revserse_embed_a = reverse_tokenizer(embedA, attA)
        return linearA_addition, linearB_addition

    def conv_bn_act(self, MLO, CC, filters, drop_out=0.0, t=0):
        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO1 = Activation('relu')(MLO)
        CC1 = Activation('relu')(CC)

        if self.args.connection_type == "Average":
            MLO, CC, A = self.Averaged_Excitation_Attention(MLO1, CC1, t=1)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO1, CC1)
        elif self.args.connection_type == "Add":
            MLO, CC,  A = self.Addition_Excitation_Attention(MLO1, CC1, t=1)
        MLO = Add()([MLO, MLO1])
        CC = Add()([CC, CC1])
        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO = Activation('relu')(MLO)
        CC = Activation('relu')(CC)
        if t == 0:
            A = 0
        MLO = Add()([MLO, MLO1])
        CC = Add()([CC, CC1])
        return MLO, CC, A

    def selective_layer(self, MLO, CC, filters, compression=0.5, drop_out=0.0, t=0):
        MLO = Conv2D(filters, (3, 3), activation=None,
                     dilation_rate=2, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None,
                    dilation_rate=2, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO1 = Activation('relu')(MLO)
        CC1 = Activation('relu')(CC)

        if self.args.connection_type == "Average":
            MLO, CC, A = self.Averaged_Excitation_Attention(MLO1, CC1, t=1)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO1, CC1)
        elif self.args.connection_type == "Add":
            MLO, CC,  A = self.Addition_Excitation_Attention(MLO1, CC1, t=1)
        MLO = Add()([MLO, MLO1])
        CC = Add()([CC, CC1])

        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)
        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO2 = Activation('relu')(MLO)
        CC2 = Activation('relu')(CC)

        MLO = Add()([MLO2, MLO1])
        CC = Add()([CC2, CC1])
        MLO = GlobalAveragePooling2D()(MLO)
        CC = GlobalAveragePooling2D()(CC)
        MLO = Dense(int(filters * compression))(MLO)
        CC = Dense(int(filters * compression))(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO = Activation('relu')(MLO)
        CC = Activation('relu')(CC)

        MLO = Dense(int(filters))(MLO)
        CC = Dense(int(filters))(CC)

        MLO3 = Activation('sigmoid')(MLO)
        CC3 = Activation('sigmoid')(CC)
        MLO4 = Lambda(lambda x: 1 - x)(MLO3)
        CC4 = Lambda(lambda x: 1 - x)(CC3)

        MLO5 = multiply([MLO1, MLO3])
        MLO6 = multiply([MLO2, MLO4])

        CC5 = multiply([CC1, CC3])
        CC6 = multiply([CC2, CC4])

        return add([MLO5, MLO6]), add([CC5, CC6]), A

    def selective_layer_de(self, MLO, CC, A, filters, compression=0.5, drop_out=0.0, t=0):
        MLO = Conv2D(filters, (3, 3), activation=None,
                     dilation_rate=2, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None,
                    dilation_rate=2, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO1 = Activation('relu')(MLO)
        CC1 = Activation('relu')(CC)

        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention_skip(
                MLO1, CC1, A, t=t)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO1, CC1)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention_skip(
                MLO1, CC1, A, t=t)
        MLO = Add()([MLO, MLO1])
        CC = Add()([CC, CC1])

        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)
        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO2 = Activation('relu')(MLO)
        CC2 = Activation('relu')(CC)

        MLO = Add()([MLO2, MLO1])
        CC = Add()([CC2, CC1])
        MLO = GlobalAveragePooling2D()(MLO)
        CC = GlobalAveragePooling2D()(CC)
        MLO = Dense(int(filters * compression))(MLO)
        CC = Dense(int(filters * compression))(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO = Activation('relu')(MLO)
        CC = Activation('relu')(CC)

        MLO = Dense(int(filters))(MLO)
        CC = Dense(int(filters))(CC)

        MLO3 = Activation('sigmoid')(MLO)
        CC3 = Activation('sigmoid')(CC)
        MLO4 = Lambda(lambda x: 1 - x)(MLO3)
        CC4 = Lambda(lambda x: 1 - x)(CC3)

        MLO5 = multiply([MLO1, MLO3])
        MLO6 = multiply([MLO2, MLO4])

        CC5 = multiply([CC1, CC3])
        CC6 = multiply([CC2, CC4])

        return add([MLO5, MLO6]), add([CC5, CC6])

    def conv_bn_act_de(self, MLO, CC, A, filters, drop_out=0.2, t=0):
        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO1 = Activation('relu')(MLO)
        CC1 = Activation('relu')(CC)

        if self.args.connection_type == "Average":
            MLO, CC = self.Averaged_Excitation_Attention_skip(
                MLO1, CC1, A, t=t)
        elif self.args.connection_type == "Self":
            MLO, CC = self.Self_Excitation_Attention(MLO1, CC1)
        elif self.args.connection_type == "Add":
            MLO, CC = self.Addition_Excitation_Attention_skip(
                MLO1, CC1, A, t=t)

        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        CC = Conv2D(filters, (3, 3), activation=None, padding='same')(CC)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
            CC = Dropout(drop_out)(CC)

        MLO = BatchNormalization()(MLO)
        CC = BatchNormalization()(CC)
        MLO = Activation('relu')(MLO)
        CC = Activation('relu')(CC)
        MLO = Add()([MLO, MLO1])
        CC = Add()([CC, CC1])

        return MLO, CC
    
    
    
    
    def conv_bn_act_single(self, MLO, filters, drop_out=0.2, t=0):
        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)
        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)
        MLO = BatchNormalization()(MLO)
        MLO1 = Activation('relu')(MLO)
        MLO = Conv2D(filters, (3, 3), activation=None, padding='same')(MLO)

        if drop_out > 0:
            MLO = Dropout(drop_out)(MLO)

        MLO = BatchNormalization()(MLO)
        MLO = Activation('relu')(MLO)
        MLO = Add()([MLO, MLO1])
        return MLO


    def unet_pooling(self, MLO, CC):
        MLO = AveragePooling2D((2, 2))(MLO)
        CC = AveragePooling2D((2, 2))(CC)
        return MLO, CC

    def bottleneck(self, MLO, CC, filters, t=0):
        MLO, CC, A = self.conv_bn_act(MLO, CC, filters)
        return MLO, CC

    def bottleneck2(self, MLO, CC, filters, drop_out=0.0):
        Combined = Add()([MLO, CC])
        Combined = Conv2D(filters, (3, 3), activation=None,
                          padding='same')(Combined)
        if drop_out > 0:
            Combined = Dropout(drop_out)(Combined)
        Combined = BatchNormalization()(Combined)
        Combined1 = Activation('relu')(Combined)

        Combined = Conv2D(filters, (3, 3), activation=None,
                          padding='same')(Combined1)
        if drop_out > 0:
            Combined = Dropout(drop_out)(Combined)
        Combined = BatchNormalization()(Combined)
        Combined = Activation('relu')(Combined)
        Combined = Add()([Combined, Combined1])
        return Combined

    def tran_dual(self, MLO, CC, filters):
        MLO = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(MLO)
        CC = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(CC)
        return MLO, CC
    def tran_dual_single(self,  CC, filters):
        CC = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(CC)
        return CC

    def unet_regression(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.connection_type == "Average":
            t = 1
        elif self.args.connection_type == "Self":
            t = 0
        elif self.args.connection_type == "Add":
            t = 1
        elif self.args.connection_type == "Baseline":
            t = 0

        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        m1, c1, A1 = self.conv_bn_act(inputA, inputB, filters, t=t)
        pm1, pc1 = self.unet_pooling(m1, c1)
        m2, c2, A2 = self.conv_bn_act(pm1, pc1, filters, t=t)

        pm2, pc2 = self.unet_pooling(m2, c2)
        m3, c3, A3 = self.conv_bn_act(pm2, pc2, filters*2, t=t)
        pm3, pc3 = self.unet_pooling(m3, c3)
        m4, c4, A4 = self.conv_bn_act(pm3, pc3, filters*4, t=t)
        pm4, pc4 = self.unet_pooling(m4, c4)

        bm, bcc = self.bottleneck(pm4, pc4, filters*8)
        bm = AveragePooling2D((7, 7))(bm)
        bcc = AveragePooling2D((7, 7))(bcc)
        dropout = Dropout(0.3)(bm)
        flattened = Flatten()(dropout)

        lc1 = Dense(128, activation="relu")(flattened)  # was 100
        lc2 = Dense(64, activation="relu")(lc1)
        lc3 = Dense(32, activation="relu")(lc2)  # was 100
        lc4 = Dense(4, activation="sigmoid")(lc3)

        dropoutc = Dropout(0.3)(bcc)
        flattenedc = Flatten()(dropoutc)

        lcc1 = Dense(128, activation="relu")(flattenedc)  # was 100
        lcc2 = Dense(64, activation="relu")(lcc1)
        lcc3 = Dense(32, activation="relu")(lcc2)  # was 100
        lcc4 = Dense(4, activation="sigmoid")(lcc3)

        #pred_output = self.Output_Block(bm, bcc)

        model = keras.models.Model(
            inputs=[inputA, inputB], outputs=[lc4, lcc4])

        return model

    def unet_dual(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.connection_type == "Average":
            t = 1
        elif self.args.connection_type == "Self":
            t = 0
        elif self.args.connection_type == "Add":
            t = 1
        elif self.args.connection_type == "Baseline":
            t = 0

        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        m1, c1, A1 = self.conv_bn_act(inputA, inputB, filters, t=t)
        pm1, pc1 = self.unet_pooling(m1, c1)
        m2, c2, A2 = self.conv_bn_act(pm1, pc1, filters*2, t=t)
        pm2, pc2 = self.unet_pooling(m2, c2)
        m3, c3, A3 = self.conv_bn_act(pm2, pc2, filters*4, t=t)
        pm3, pc3 = self.unet_pooling(m3, c3)

        bm, bcc = self.bottleneck(pm3, pc3, filters*8)

        mc2, cc2 = self.tran_dual(bm, bcc, filters*4)
        mc2 = concatenate([mc2, m3], axis=3)
        cc2 = concatenate([cc2, c3], axis=3)
        m6, c6 = self.conv_bn_act_de(mc2, cc2, A3, filters*4, t=t)

        mc3, cc3 = self.tran_dual(m6, c6, filters*2)
        mc3 = concatenate([mc3, m2], axis=3)
        cc3 = concatenate([cc3, c2], axis=3)
        m7, c7 = self.conv_bn_act_de(mc3, cc3, A2, filters*2, t=t)

        mc4, cc4 = self.tran_dual(m7, c7, filters)
        mc4 = concatenate([mc4, m1], axis=3)
        cc4 = concatenate([cc4, c1], axis=3)
        m8, c8 = self.conv_bn_act_de(mc4, cc4, A1, filters, t=t)
        if self.args.num_pool > 0:
            m9 = Conv2D(self.args.num_pool, (1, 1),
                        padding="same", activation='sigmoid')(m8)
            c9 = Conv2D(self.args.num_pool, (1, 1),
                        padding="same", activation='sigmoid')(c8)
            m9_max = tf.reduce_max(m9, axis=3)
            c9_max = tf.reduce_max(c9, axis=3)
            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m9_max, c9_max])
        else:
            m9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(m8)
            c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)

            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m9, c9])
        return model

    def unet_dual2(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.connection_type == "Average":
            t = 1
        elif self.args.connection_type == "Self":
            t = 0
        elif self.args.connection_type == "Add":
            t = 1
        elif self.args.connection_type == "Baseline":
            t = 0

        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        m1, c1, A1 = self.conv_bn_act(inputA, inputB, filters, t=t)
        pm1, pc1 = self.unet_pooling(m1, c1)
        m2, c2, A2 = self.conv_bn_act(pm1, pc1, filters, t=t)
        pm2, pc2 = self.unet_pooling(m2, c2)
        m3, c3, A3 = self.conv_bn_act(pm2, pc2, filters*2, t=t)
        pm3, pc3 = self.unet_pooling(m3, c3)
        m4, c4, A4 = self.conv_bn_act(pm3, pc3, filters*4, t=t)
        pm4, pc4 = self.unet_pooling(m4, c4)
        m5, c5, A5 = self.conv_bn_act(pm4, pc4, filters*8, t=t)
        pm5, pc5 = self.unet_pooling(m5, c5)

        Combined = Add()([pm5, pc5])

        dropout = Dropout(0.3)(Combined)
        flattened = Flatten()(dropout)
        BB1 = Dense(flattened.shape[-1], activation='linear')(flattened)
        BB2 = Dense(flattened.shape[-1], activation='linear')(BB1)
        Reshape = tf.reshape(
            BB2, [-1, Combined.shape[1], Combined.shape[2], Combined.shape[3]])
        #bcombined = self.bottleneck2(pm5, pc5, filters*32)

        mc6, cc6 = self.tran_dual(Reshape, Reshape, filters*8)
        mc6 = concatenate([mc6, m5], axis=3)
        cc6 = concatenate([cc6, c5], axis=3)
        m6, c6 = self.conv_bn_act_de(mc6, cc6, A5, filters*8, t=t)

        mc7, cc7 = self.tran_dual(m6, c6, filters*4)
        mc7 = concatenate([mc7, m4], axis=3)
        cc7 = concatenate([cc7, c4], axis=3)
        m7, c7 = self.conv_bn_act_de(mc7, cc7, A4, filters*4, t=t)

        mc8, cc8 = self.tran_dual(m7, c7, filters*2)
        mc8 = concatenate([mc8, m3], axis=3)
        cc8 = concatenate([cc8, c3], axis=3)
        m8, c8 = self.conv_bn_act_de(mc8, cc8, A3, filters*2, t=t)

        mc9, cc9 = self.tran_dual(m8, c8, filters)
        mc9 = concatenate([mc9, m2], axis=3)
        cc9 = concatenate([cc9, c2], axis=3)
        m9, c9 = self.conv_bn_act_de(mc9, cc9, A2, filters, t=t)

        mc10, cc10 = self.tran_dual(m9, c9, filters)
        mc10 = concatenate([mc10, m1], axis=3)
        cc10 = concatenate([cc10, c1], axis=3)
        m10, c10 = self.conv_bn_act_de(mc10, cc10, A1, filters, t=t)
        if self.args.num_pool > 0:
            m11 = Conv2D(self.args.num_pool, (1, 1),
                         padding="same", activation='sigmoid')(m10)
            c11 = Conv2D(self.args.num_pool, (1, 1),
                         padding="same", activation='sigmoid')(c10)
            m11_max = tf.reduce_max(m11, axis=3)
            c11_max = tf.reduce_max(c11, axis=3)
            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m11_max, c11_max])
        else:
            m11 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(m10)
            c11 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c10)

            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m11, c11])
        return model
    
    def unet_dual3(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.connection_type == "Average":
            t = 1
        elif self.args.connection_type == "Self":
            t = 0
        elif self.args.connection_type == "Add":
            t = 1
        elif self.args.connection_type == "Baseline":
            t = 0

        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        m1, c1, A1 = self.conv_bn_act(inputA, inputB, filters, t=t)
        pm1, pc1 = self.unet_pooling(m1, c1)
        m2, c2, A2 = self.conv_bn_act(pm1, pc1, filters, t=t)
        pm2, pc2 = self.unet_pooling(m2, c2)
        m3, c3, A3 = self.conv_bn_act(pm2, pc2, filters*2, t=t)
        pm3, pc3 = self.unet_pooling(m3, c3)
        m4, c4, A4 = self.conv_bn_act(pm3, pc3, filters*4, t=t)
        pm4, pc4 = self.unet_pooling(m4, c4)
        m5, c5, A5 = self.conv_bn_act(pm4, pc4, filters*8, t=t)
        pm5, pc5 = self.unet_pooling(m5, c5)

        Combined = Add()([pm5, pc5])
        #Combined = keras.layers.Subtract()([pm5, pc5])
        dropout = Dropout(0.3)(Combined)
        flattened = Flatten()(dropout)
        BB1 = Dense(flattened.shape[-1], activation='linear')(flattened)
        BB2 = Dense(flattened.shape[-1], activation='linear')(BB1)
        Reshape = tf.reshape(
            BB2, [-1, Combined.shape[1], Combined.shape[2], Combined.shape[3]])
        #bcombined = self.bottleneck2(pm5, pc5, filters*32)

        mc6 = self.tran_dual_single(Reshape, filters*8)
        mc6 = concatenate([mc6, m5], axis=3)
        m6 = self.conv_bn_act_single(mc6, filters*8, t=t)

        mc7= self.tran_dual_single(m6, filters*4)
        mc7 = concatenate([mc7, m4], axis=3)
        m7= self.conv_bn_act_single(mc7, filters*4, t=t)

        mc8 = self.tran_dual_single(m7, filters*2)
        mc8 = concatenate([mc8, m3], axis=3)
        m8 = self.conv_bn_act_single(mc8, filters*2, t=t)
        mc9 = self.tran_dual_single(m8, filters)
        mc9 = concatenate([mc9, m2], axis=3)

        m9 = self.conv_bn_act_single(mc9, filters, t=t)

        mc10 = self.tran_dual_single(m9,  filters)
        mc10 = concatenate([mc10, m1], axis=3)
        m10 = self.conv_bn_act_single(mc10, filters, t=t)

        if self.args.num_pool > 0:
            m11 = Conv2D(self.args.num_pool, (1, 1),
                         padding="same", activation='sigmoid')(m10)
            #c11 = Conv2D(self.args.num_pool, (1, 1),padding="same", activation='sigmoid')(c10)
            m11_max = tf.reduce_max(m11, axis=3)
            #c11_max = tf.reduce_max(c11, axis=3)
            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m11_max])
        else:
            m11 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(m10)
            #c11 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c10)

            model = keras.models.Model(
                inputs=[inputA, inputB], outputs=[m11])
        return model

    def Unet_class_model(self):
        mod = self.unet_dual2()
        mod.load_weights(self.args.weight_path)
        if self.args.connection_type == "Average":
            Extract = Model(inputs=mod.input, outputs=mod.layers[141].output)
        elif self.args.connection_type == "Baseline":
            Extract = Model(inputs=mod.input, outputs=mod.layers[96].output)
        out = Dense(self.args.num_output_classes,
                    activation="softmax")(Extract)
        model = keras.models.Model(inputs=Extract.inputs, outputs=out)
        return mod

    def skunet_dual(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.connection_type == "Average":
            t = 1
        elif self.args.connection_type == "Self":
            t = 0
        elif self.args.connection_type == "Add":
            t = 1
        elif self.args.connection_type == "Baseline":
            t = 0

        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        m1, c1, A1 = self.selective_layer(inputA, inputB, filters, t=t)
        pm1, pc1 = self.unet_pooling(m1, c1)
        m2, c2, A2 = self.selective_layer(pm1, pc1, filters*2, t=t)
        pm2, pc2 = self.unet_pooling(m2, c2)
        m3, c3, A3 = self.selective_layer(pm2, pc2, filters*4, t=t)
        pm3, pc3 = self.unet_pooling(m3, c3)

        bm, bcc = self.bottleneck(pm3, pc3, filters*8)

        mc2, cc2 = self.tran_dual(bm, bcc, filters*4)
        mc2 = concatenate([mc2, m3], axis=3)
        cc2 = concatenate([cc2, c3], axis=3)
        m6, c6 = self.selective_layer_de(mc2, cc2, A3, filters*4, t=t)

        mc3, cc3 = self.tran_dual(m6, c6, filters*2)
        mc3 = concatenate([mc3, m2], axis=3)
        cc3 = concatenate([cc3, c2], axis=3)
        m7, c7 = self.conv_bn_act_de(mc3, cc3, A2, filters*2, t=t)

        mc4, cc4 = self.tran_dual(m7, c7, filters)
        mc4 = concatenate([mc4, m1], axis=3)
        cc4 = concatenate([cc4, c1], axis=3)
        m8, c8 = self.selective_layer_de(mc4, cc4, A1, filters, t=t)

        m9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(m8)
        c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)

        model = keras.models.Model(inputs=[inputA, inputB], outputs=[m9, c9])

        return model

    def unet(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        c1 = self.conv_bn_act_orig(inputA, filters)
        c1 = self.conv_bn_act_orig(c1, filters)
        p1 = MaxPooling2D((2, 2))(c1)
        filters = 2 * filters

        c2 = self.conv_bn_act_orig(p1, filters)
        c2 = self.conv_bn_act_orig(c2, filters)
        p2 = MaxPooling2D((2, 2))(c2)
        filters = 2 * filters

        c3 = self.conv_bn_act_orig(p2, filters)
        c3 = self.conv_bn_act_orig(c3, filters)
        p3 = MaxPooling2D((2, 2))(c3)
        filters = 2 * filters

        cm = self.conv_bn_act_orig(p3, filters)
        cm = self.conv_bn_act_orig(cm, filters)

        filters = filters // 2

        u3 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(cm)

        u3 = concatenate([u3, c3], axis=3)

        c6 = self.conv_bn_act_orig(u3, filters)
        c6 = self.conv_bn_act_orig(c6, filters)

        filters = filters // 2

        u2 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(c6)
        u2 = concatenate([u2, c2], axis=3)

        c7 = self.conv_bn_act_orig(u2, filters)
        c7 = self.conv_bn_act_orig(c7, filters)

        filters = filters // 2

        u1 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(c7)

        u1 = concatenate([u1, c1], axis=3)

        c8 = self.conv_bn_act_orig(u1, filters)
        c8 = self.conv_bn_act_orig(c8, filters)
        if self.args.num_pool > 0:
            c9 = Conv2D(self.args.num_pool, (1, 1),
                        padding="same", activation='sigmoid')(c8)

            c9_max = tf.reduce_max(c9, axis=3)
            model = keras.models.Model(inputs=inputA, outputs=c9_max)
        else:
            c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)
            model = keras.models.Model(inputs=inputA, outputs=c9)

        return model

    def unet_decon(self, filters=16):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size,
                       self.args.img_size, self.N_channels))

        c1 = self.conv_bn_act_depth(inputA, filters)
        c1 = self.conv_bn_act_depth(c1, filters)
        p1 = MaxPooling2D((2, 2))(c1)
        filters = 2 * filters

        c2 = self.conv_bn_act_depth(p1, filters)
        c2 = self.conv_bn_act_depth(c2, filters)
        p2 = MaxPooling2D((2, 2))(c2)
        filters = 2 * filters

        c3 = self.conv_bn_act_depth(p2, filters)
        c3 = self.conv_bn_act_depth(c3, filters)
        p3 = MaxPooling2D((2, 2))(c3)
        filters = 2 * filters

        cm = self.conv_bn_act_depth(p3, filters)
        cm = self.conv_bn_act_depth(cm, filters)

        filters = filters // 2

        u3 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(cm)

        u3 = concatenate([u3, c3], axis=3)

        c6 = self.conv_bn_act_depth(u3, filters)
        c6 = self.conv_bn_act_depth(c6, filters)

        filters = filters // 2

        u2 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(c6)
        u2 = concatenate([u2, c2], axis=3)

        c7 = self.conv_bn_act_depth(u2, filters)
        c7 = self.conv_bn_act_depth(c7, filters)

        filters = filters // 2

        u1 = Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same')(c7)

        u1 = concatenate([u1, c1], axis=3)

        c8 = self.conv_bn_act_depth(u1, filters)
        c8 = self.conv_bn_act_depth(c8, filters)
        if self.args.num_pool > 0:
            c9 = Conv2D(self.args.num_pool, (1, 1),
                        padding="same", activation='sigmoid')(c8)

            c9_max = tf.reduce_max(c9, axis=3)
            model = keras.models.Model(inputs=inputA, outputs=c9_max)
        else:
            c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)
            model = keras.models.Model(inputs=inputA, outputs=c9)

        return model

    def conv_bn_act_orig(self, x, filters, drop_out=0.0):
        x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

        if drop_out > 0:
            x = Dropout(drop_out)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def conv_bn_act_depth(self, x, filters, drop_out=0.0):
        x = DepthwiseConv2D((3, 3), activation=None, padding='same')(x)

        if drop_out > 0:
            x = Dropout(drop_out)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
