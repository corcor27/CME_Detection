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
from Model_Optimizer import CosineAnnealingScheduler, CustomSchedule
from tensorflow.keras.utils import Progbar
#from models_development import  DualResNet50, DualAttResNet50, Crossview2DualAttResNet50
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score, rand_score, adjusted_mutual_info_score, mutual_info_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Models import Create_Model
from tensorflow.keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow_addons as tfa
from skimage.measure import label, regionprops
import pandas as pd
import umap
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image
from math import sqrt
from sklearn.metrics import roc_auc_score
#import sys

np_config.enable_numpy_behavior()

formatter = logging.Formatter(
    '%(asctime)s - (%(name)s) %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


class CustomModel:
    def __init__(self, args):
        self.args = args
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        

    def Create_Model_From_args(self):
        if self.args.model_type == 'Classification':
            model = Create_Model(self.args).Create_Classification_Model()

        elif self.args.model_type == 'Segmentation':
            model = Create_Model(self.args).Create_Segmentation_Model()
        return model

    def Load_model_weight(self):
        if self.args.weight_path == 'None':
            # self.model.load_weights(self.weight_path)
            print("wieghts loading skipped")
        else:
            if self.args.load_weight:
                print("Wieghts loaded from {}".format(self.args.weight_path))
                self.model.load_weights(self.args.weight_path)
            else:
                print("wieghts loading skipped")

    def Save_model_weight(self):
        if self.args.wieghts_output_dir == 'None':
            print("No Weights path specified")
        else:
            if self.args.model_save:
                self.model.save(self.args.wieghts_output_dir, save_format='h5')
            else:
                self.model.save_weights(self.args.wieghts_output_dir)
                print("Wieghts are definetly saved")
    
    def iou_tensorflow(self, b1, b2):
        b2 = tf.convert_to_tensor(b2)
        if not b2.dtype.is_floating:
            b2 = tf.cast(b2, tf.float32)
        b1 = tf.cast(b1, b2.dtype)
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        iou = tf.squeeze(iou)
        return iou

    def define_optimizer(self):
        if self.args.Optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.args.Starting_lr)
        elif self.args.Optimizer == "SGD":
            return tf.keras.optimizers.experimental.SGD(learning_rate=self.args.Starting_lr)

    def loss_function(self):
        if self.args.model_type == 'Classification':
            if self.args.num_output_classes == 2:
                if self.args.loss_type == "cross":
                    print("loss is good")
                    return keras.losses.BinaryCrossentropy()
                    
                elif self.args.loss_type == "mse":
                    return tf.keras.losses.MeanSquaredError()
                elif self.args.loss_type == "focal":
                    return tf.keras.losses.BinaryFocalCrossentropy()
                elif self.args.loss_type == "sparse":
                    return tf.nn.sigmoid_cross_entropy_with_logits()
                
                    print("loss is good")
                    
            else:
                #return tf.keras.losses.MeanSquaredError()
                #return DiceLoss_Multiclass(self.args)
                #return IOULoss2(self.args)
                #return keras.losses.BinaryCrossentropy()
                #return DICE_Regression_Loss(self.args)
                return Focal_DICE_Regression_Loss(self.args)
            
        elif self.args.model_type == 'Segmentation':
            if self.args.training_type == "Regression":
                return RegressionLoss()
            else:
                return DiceLoss(self.args)

    def Load_Lr_Schedule(self):
        if self.args.lr_schedule == "Cosine":
            # T_max=self.args.Max_lr, eta_max=5e-3, warmup_epochs = 10, eta_min=1e-7
            return CosineAnnealingScheduler(self.args)
        elif self.args.lr_schedule == "Exp_decay":
            # patience=2,factor=0.00001,verbose=1, optim_lr=self.optimizer.learning_rate, reduce_lin=True
            return CustomSchedule(args = self.args, patience=3,factor=0.9,verbose=1, optim_lr=self.optimizer.learning_rate, reduce_lin=True, mode="max")
    def Prep_training(self, Train_data, Validation_data, test_generator=None):
        
        self.optimizer = self.define_optimizer()
        self.loss_fn = self.loss_function()
        self.Schedule = self.Load_Lr_Schedule()
        if self.args.model_type == 'Classification':
            if self.args.training_type != "Regression":
                if self.args.num_output_classes > 2 and self.args.num_output_objects == 1:
                    print("training is all good")
                    self.train_acc_metric = keras.metrics.CategoricalAccuracy()
                    self.val_acc_metric = keras.metrics.CategoricalAccuracy()
                elif self.args.num_output_classes < 3 and self.args.num_output_objects == 1:
                    self.train_acc_metric = keras.metrics.BinaryAccuracy()
                    self.val_acc_metric = keras.metrics.BinaryAccuracy()
                    print("binary acc being used")
                else:
                    self.train_acc_metric = keras.metrics.CategoricalAccuracy()
                    self.val_acc_metric = keras.metrics.CategoricalAccuracy()
            else:
                self.train_acc_metric = tf.keras.metrics.MeanMetricWrapper(fn=self.iou_tensorflow)
                self.val_acc_metric = tf.keras.metrics.MeanMetricWrapper(fn=self.iou_tensorflow) 
                #self.train_acc_metric = tf.keras.metrics.Accuracy()
                #self.val_acc_metric = tf.keras.metrics.Accuracy()

        elif self.args.model_type == 'Segmentation':
            if self.args.training_type == "Regression":
                self.train_acc_metric = tf.keras.metrics.Accuracy()
                self.val_acc_metric = tf.keras.metrics.Accuracy()
            else:
                self.train_acc_metric = tf.keras.metrics.MeanIoU(num_classes=2)
                self.val_acc_metric = tf.keras.metrics.MeanIoU(num_classes=2)
        self.Train_data = Train_data
        self.Validation_data = Validation_data
        if self.args.features:
            self.test_data = test_generator

        self.Train_acc = np.zeros(
            shape=(self.args.max_epochs,), dtype=np.float32)
        self.Val_acc = np.zeros(
            shape=(self.args.max_epochs,), dtype=np.float32)
        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        
        if self.args.load_pretain == False:
            self.model = self.Create_Model_From_args()
        else:
            self.model = self.Create_Model_From_args()
            self.Load_model_weight()
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.train_acc_metric)
        c_count = self.Train_data.Class_Count()
        self.Class_weights = c_count
        fac_c = 1 * np.sum(c_count)
        for ind in range(len(c_count)):
            score = math.log(fac_c/float(c_count[ind]))
            self.Class_weights[ind] = score if score > 1.0 else 1.0
            #self.Class_weights = np.sum(c_count) / c_count
            #self.Class_weights = self.Class_weights / np.sum(self.Class_weights)
            
    def Prep_training_att(self, Train_data, Validation_data, test_generator=None):
        self.Train_data = Train_data
        self.Validation_data = Validation_data

    @tf.function
    def train_step(self, material, labels):
        with tf.GradientTape() as tape:
            if len(labels) == 1:
                y_pred = self.model(material, training=True)
                loss_value = self.loss_fn(labels[0], y_pred)
                self.train_acc_metric.update_state(labels[0], y_pred)
                if self.args.class_weights:
                    loss_value = tf.reduce_mean(self.class_weights*loss_value)
            else:
                y_pred = self.model(material, training=True)
                loss_value = self.loss_fn(labels, y_pred)
                self.train_acc_metric.update_state(labels, y_pred)
                if self.args.class_weights:
                    loss_value = tf.reduce_mean(self.class_weights*loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return loss_value, y_pred

    @tf.function
    def train_step_regression(self,material, labels):
        with tf.GradientTape() as tape:
            if len(labels) == 1:
                y_pred = self.model(material, training=True)
                loss_value = self.loss_fn(labels[0], y_pred)
                self.train_acc_metric.update_state(labels[0], y_pred)
            else:
                y_pred = self.model(material, training=True)
                loss_value = self.loss_fn(labels, y_pred)
                self.train_acc_metric.update_state(labels, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        
        return loss_value, y_pred

    @tf.function
    def test_step_regression(self, material, labels):
        if len(labels) == 1:
            y_pred = self.model(material, training=True)
            loss_value = self.loss_fn(labels[0], y_pred)
            self.val_acc_metric.update_state(labels[0], y_pred)
        else:
            y_pred = self.model(material, training=True)
            loss_value = self.loss_fn(labels, y_pred)
            self.val_acc_metric.update_state(labels, y_pred)
        
        return loss_value

    @tf.function
    def test_step(self, material, labels):
        if len(labels) == 1:
            y_pred = self.model(material, training=False)
            loss_value = self.loss_fn(labels[0], y_pred)
            loss_value = tf.reduce_mean(self.class_weights*loss_value)
            self.val_acc_metric.update_state(labels[0], y_pred)
        else:
            y_pred = self.model(material, training=False)
            loss_value = self.loss_fn(labels, y_pred)
            loss_value = tf.reduce_mean(self.class_weights*loss_value)
            self.val_acc_metric.update_state(labels, y_pred)
        return loss_value

    @tf.function
    def __predict(self, x):
        val_pred = self.model(x, training=False)
        return val_pred


    def plot_confusion(self, Val_PRED, Val_GRON):
        f, axes = plt.subplots(1, 2, figsize=(25, 15))
        axes = axes.ravel()
        for i in range(0, len(self.args.prediction_classes)):
            disp = ConfusionMatrixDisplay(confusion_matrix(
                Val_GRON[:, i], Val_PRED[:, i]), display_labels=[0, 1])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title('{}'.format(self.args.prediction_classes[i]))
            if i < 10:
                disp.ax_.set_xlabel('Predition Label')
            if i % 2 != 0:
                disp.ax_.set_ylabel('True Label')
            disp.im_.colorbar.remove()

        plt.subplots_adjust(wspace=0.30, hspace=0.1)
        plt.rcParams['font.size'] = 60
        if self.args.mode != "fold_testing":
            plt.suptitle("Model:{} Fold:{} Image Size:{} ".format(
                self.args.connection_type, self.args.test_fold, self.args.img_size))
        else:
            plt.suptitle("Model:{} Image Size:{} Testset:{}".format(
                self.args.connection_type, self.args.img_size, self.args.analysis_type))
        f.colorbar(disp.im_, ax=axes)
        if self.args.mode != "fold_testing":
            p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "CC{}_VAL{}_TEST{}_RE{}_IMG{}.png".format(
                self.args.connection_type, self.args.val_fold, self.args.test_fold, self.args.repeat, self.args.img_size))
        else:
            p = os.path.join(self.args.post_analysis_folder, "{}_{}_{}_{}.png".format(
                self.args.connection_type, self.args.img_size, self.args.analysis_type, self.args.backbone))
        plt.savefig(p)

        return None

    def collect_imgs(self):
        q = -1

        if self.args.num_input_objects == 1:

            for x_batch_val,  y_batch_val in self.Validation_data:

                val_pred = np.array(self.__predict(x_batch_val))
                y_pred = self.__predict(x_batch_val)
                #val_pred = (val_pred >= 0.5).astype(np.uint8)
                
                val_pred = np.squeeze(val_pred)
                val_ground = np.squeeze(y_batch_val)
                #cv2.imwrite("DATASET/temp_folder/{}_L.png".format(q), val_pred[:,:])
                #cv2.imwrite("DATASET/temp_folder/{}_R.png".format(q), val_pred[1,:,:])
                #self.val_acc_metric.update_state(y_batch_val, val_pred)
                plt.imshow(val_ground*255)
                q += 1
                break

        else:
            for x_batch_val, z_batch_val, y_batch_val, w_batch_val in self.Validation_data:
                q += 1

                val_pred = np.array(self.model(
                    [x_batch_val,  z_batch_val], training=False))
                y_pred = self.model(
                    [x_batch_val,  z_batch_val], training=False)
                val_pred = (val_pred >= 0.5).astype(np.uint8)

                val_pred = np.squeeze(val_pred)
                y_batch_val = np.squeeze(y_batch_val)
                w_batch_val = np.squeeze(w_batch_val)
                x_batch_val = np.squeeze(x_batch_val)
                z_batch_val = np.squeeze(z_batch_val)
                #print(val_pred.shape, val_ground.shape)

                self.val_acc_metric.update_state(
                    [y_batch_val, w_batch_val], val_pred)
                if q == -2:
                    left_pos = np.where(y_batch_val == 1)
                    right_pos = np.where(w_batch_val == 1)
                    lz1, lz2, lx1, lx2 = np.min(left_pos[0]), np.max(
                        left_pos[0]), np.min(left_pos[1]), np.max(left_pos[1])
                    rz1, rz2, rx1, rx2 = np.min(right_pos[0]), np.max(
                        right_pos[0]), np.min(right_pos[1]), np.max(right_pos[1])
                    label_image_left = label(val_pred[0, :, :])
                    label_image_right = label(val_pred[1, :, :])

                    if np.max(label_image_left) > 0:
                        try:
                            for val in range(1, np.max(label_image_left)+1):
                                mask_pos = np.where(label_image_left == val)
                                z1, z2, x1, x2 = np.min(mask_pos[0]), np.max(
                                    mask_pos[0]), np.min(mask_pos[1]), np.max(mask_pos[1])
                                mid_pointz, mid_pointx = int(round(((np.max(mask_pos[0]) - np.min(mask_pos[0]))/2) + np.min(
                                    mask_pos[0]), 0)), int(round(((np.max(mask_pos[1]) - np.min(mask_pos[1]))/2) + np.min(mask_pos[1]), 0))

                                if lz1 < mid_pointz < lz2:
                                    if lx1 < mid_pointx < lx2:

                                        IMG_PATH1 = os.path.join(self.args.dataset_path_img, str(
                                            self.Validation_data.get_name_dataframe(self.args.image_col, q)) + self.args.ext_type)
                                        BASE_IMG = cv2.imread(IMG_PATH1)
                                        MASK_GROUND = cv2.resize(
                                            y_batch_val*255, (BASE_IMG.shape[1], BASE_IMG.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
                                        # plt.imshow(MASK_GROUND)
                                        MASK_PRED = np.where(
                                            label_image_left == val, 255, 0)

                                        MASK_PRED.astype(np.uint8)
                                        MASK_PRED = cv2.resize(
                                            MASK_PRED, (BASE_IMG.shape[1], BASE_IMG.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)

                                        left_pos = np.where(MASK_GROUND > 250)
                                        mask_pos = np.where(MASK_PRED > 250)
                                        lz1, lz2, lx1, lx2 = np.min(left_pos[0]), np.max(
                                            left_pos[0]), np.min(left_pos[1]), np.max(left_pos[1])
                                        z1, z2, x1, x2 = np.min(mask_pos[0]), np.max(
                                            mask_pos[0]), np.min(mask_pos[1]), np.max(mask_pos[1])
                                        IMG2 = cv2.rectangle(
                                            BASE_IMG, (x1, z1), (x2, z2), (55, 126, 184), 8)
                                        IMG2 = cv2.rectangle(
                                            IMG2, (lx1, lz1), (lx2, lz2), (255, 127, 0), 8)
                                        print(self.Validation_data.get_name_dataframe(
                                            self.args.image_col, q))
                                        cv2.imwrite("DATASET/temp_folder/{}.png".format(
                                            self.Validation_data.get_name_dataframe(self.args.image_col, q)), IMG2)
                        except:

                            continue
                    elif np.max(label_image_right) > 0:
                        try:
                            for val in range(1, np.max(label_image_right)+1):
                                mask_pos = np.where(label_image_right == val)
                                z1, z2, x1, x2 = np.min(mask_pos[0]), np.max(
                                    mask_pos[0]), np.min(mask_pos[1]), np.max(mask_pos[1])
                                mid_pointz, mid_pointx = int(round(((np.max(mask_pos[0]) - np.min(mask_pos[0]))/2) + np.min(
                                    mask_pos[0]), 0)), int(round(((np.max(mask_pos[1]) - np.min(mask_pos[1]))/2) + np.min(mask_pos[1]), 0))
                                if rz1 < mid_pointz < rz2:
                                    if rx1 < mid_pointx < rx2:
                                        IMG_PATH1 = os.path.join(self.args.dataset_path_img, str(
                                            self.Validation_data.get_name_dataframe(self.args.second_image_col, q)) + self.args.ext_type)
                                        BASE_IMG = cv2.imread(IMG_PATH1)
                                        MASK_GROUND = cv2.resize(
                                            w_batch_val*255, (BASE_IMG.shape[1], BASE_IMG.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)

                                        MASK_PRED = np.where(
                                            label_image_right == val, 255, 0)
                                        MASK_PRED.astype(np.uint8)
                                        MASK_PRED = cv2.resize(
                                            MASK_PRED, (BASE_IMG.shape[1], BASE_IMG.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
                                        left_pos = np.where(MASK_GROUND > 250)
                                        mask_pos = np.where(MASK_PRED > 250)
                                        rz1, rz2, rx1, rx2 = np.min(right_pos[0]), np.max(
                                            right_pos[0]), np.min(right_pos[1]), np.max(right_pos[1])
                                        z1, z2, x1, x2 = np.min(mask_pos[0]), np.max(
                                            mask_pos[0]), np.min(mask_pos[1]), np.max(mask_pos[1])

                                        IMG2 = cv2.rectangle(
                                            BASE_IMG, (x1, z1), (x2, z2), (55, 126, 184), 8)
                                        IMG2 = cv2.rectangle(
                                            IMG2, (rx1, rz1), (rx2, rz2), (255, 127, 0), 8)
                                        print(self.Validation_data.get_name_dataframe(
                                            self.args.second_image_col, q))
                                        cv2.imwrite("DATASET/temp_folder/{}.png".format(
                                            self.Validation_data.get_name_dataframe(self.args.second_image_col, q)), IMG2)
                        except:

                            continue
                    #
                    #

        #Val_PRED = np.squeeze(np.array(Val_PRED))
        #Val_GRON = np.squeeze(np.array(Val_GRON))
        print(self.val_acc_metric.result())

        return q

    def Extract_features_class(self):

        Name = []
        Features = []
        if self.args.connection_type == "Average":
            self.Extract = Model(
                self.model.inputs, self.model.layers[-2].output)
        elif self.args.connection_type == "Baseline":
            self.Extract = Model(
                self.model.inputs, self.model.layers[-2].output)
        df = self.Validation_data.get_dataframe()
        q = 0
        for x_batch_val, y_batch_val, z_batch_val in self.Validation_data:
            print(x_batch.shape, y_batch.shape, z_batch.shape)
            val_pred = self.Extract([x_batch_val, z_batch_val], training=False)
            val_pred = np.squeeze(val_pred)
            Features.append(val_pred)

            Name.append(df["LesionID"].iloc[q])
            q += 1
            #val_pred = np.array(self.__predict(x_batch_val))
            #y_pred = self.__predict(x_batch_val)
        Features = np.array(Features)
        print(Features.shape)
        clusterable_embedding = umap.UMAP()
        cluster = clusterable_embedding.fit_transform(Features)
        pca = PCA(n_components=2)
        cluster2 = pca.fit_transform(Features)

        df["X_{}_UMAP_POSITIONS".format(
            self.args.connection_type)] = cluster[:, 0]
        df["Y_{}_UMAP_POSITIONS".format(
            self.args.connection_type)] = cluster[:, 1]

        df["X_{}_PCA_POSITIONS".format(
            self.args.connection_type)] = cluster2[:, 0]
        df["Y_{}_PCA_POSITIONS".format(
            self.args.connection_type)] = cluster2[:, 1]

        #p = os.path.join(self.args.post_analysis_folder, "{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.img_size, self.args.analysis_type, self.args.backbone))
        #np.save(p, Features)

        df.to_excel("{}_{}_{}_{}.xlsx".format(self.args.connection_type,
                    self.args.test_fold, self.args.repeat, self.args.img_size))
        return val_pred

    def Extract_features(self, epoch=None):
        print("Running feature Extraction")
        Features = []
        Prediction = []
        #self.model.summary()
        if self.args.connection_type == "Average" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[141].output)
        elif self.args.connection_type == "Baseline" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[96].output)
        else:
            self.Extract = Model(
                self.model.inputs, self.model.layers[-2].output)
        if self.args.features == False:
            self.df = self.Validation_data.get_dataframe()
        q = 0

        #for x_batch_val, y_batch_val, z_batch_val, u_batch_val, w_batch_val in self.Validation_data:
        for x_batch_val, y_batch_val in self.Validation_data:
        #for x_batch_val, y_batch_val in self.Validation_data:
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], n_batch_val])
            #val_pred = self.Extract([x_batch_val], training=False)
            val_pred = self.Extract([x_batch_val], training=False)
            val_pred = np.squeeze(val_pred)

            Features.append(val_pred)
            
            
            
            #val_pred = np.array(self.__predict(x_batch_val))
            #y_pred = self.__predict(x_batch_val)
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val]])
            y_pred = self.__predict([x_batch_val])
            y_pred = np.array(y_pred)
            
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1
            
            Prediction.append(array)
            
            q += 1
            
        Features = np.array(Features)
        Prediction = np.array(Prediction)
        print(Features.shape)
        #np.save("{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Features)
        #np.save("{}_{}_{}_{}_names.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Name)
        clusterable_embedding = umap.UMAP()
        cluster = clusterable_embedding.fit_transform(Features)
        pca = PCA(n_components=2)
        cluster2 = pca.fit_transform(Features)
        if self.args.features:
            self.df["X_{}_UMAP_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster[:, 0]
            self.df["Y_{}_UMAP_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster[:, 1]

            self.df["X_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 0]
            self.df["Y_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 1]
        else:
            self.df["X_{}_UMAP_POSITIONS".format(
                self.args.connection_type)] = cluster[:, 0]
            self.df["Y_{}_UMAP_POSITIONS".format(
                self.args.connection_type)] = cluster[:, 1]

            self.df["X_{}_PCA_POSITIONS".format(
                self.args.connection_type)] = cluster2[:, 0]
            self.df["Y_{}_PCA_POSITIONS".format(
                self.args.connection_type)] = cluster2[:, 1]

        #p = os.path.join(self.args.post_analysis_folder, "{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.img_size, self.args.analysis_type, self.args.backbone))
        #np.save(p, Features)
        count = 0
        obj = [self.args.prediction_classes[0], self.args.prediction_classes[1]]
        for item in obj:
            item2 = item + "_pred"
            self.df[item2] = Prediction[:,count]
            df2 = self.df[(self.df[item2] == 1)]
            print(df2.shape)
            plt.scatter(df2["X_{}_UMAP_POSITIONS".format(self.args.connection_type)],
                        df2["Y_{}_UMAP_POSITIONS".format(self.args.connection_type)], label=item)
            count += 1
        #plt.legend()
        plt.title("Umap")
        plt.show()
        file_p1 = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}_UMAP.png".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        plt.savefig(file_p1)
        plt.close()
        for item in obj:
            item2 = item + "_pred"
            df2 = self.df[(self.df[item2] == 1)]
            
            plt.scatter(df2["X_{}_PCA_POSITIONS".format(self.args.connection_type)],
                        df2["Y_{}_PCA_POSITIONS".format(self.args.connection_type)], label=item)
        #plt.legend()
        plt.title("Pca")
        file_p2 = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}_PCA.png".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        plt.savefig(file_p2)
        plt.close()
        p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}_{}.xlsx".format(self.args.connection_type,self.args.test_fold, self.args.repeat, self.args.img_size, self.args.tag))
        self.df.to_excel(p)
        return None
    
    
    
    def Extract_features_model(self, epoch=None, threshold = 0.9):
        print("Running feature Extraction")
        Features = []
        Prediction = []
        set_belong = []
        if self.args.connection_type == "Average" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[141].output)
        elif self.args.connection_type == "Baseline" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[96].output)
        else:
            self.Extract = Model(
                self.model.inputs, self.model.layers[-2].output)
        if self.args.features == False:
            self.df = self.Train_data.get_dataframe()
            self.df_val = self.Validation_data.get_dataframe()
        q = 0
        self.df_combined = pd.concat([self.df, self.df_val], axis=0)
        
        # for x_batch_val, y_batch_val, z_batch_val, u_batch_val, w_batch_val, n_batch_val, q_batch_val in self.Validation_data:
        for x_batch_val, y_batch_val in self.Train_data:
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], n_batch_val])
            val_pred = self.Extract([x_batch_val], training=False)
            #val_pred = self.Extract([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], [n_batch_val, q_batch_val]], training=False)
            val_pred = np.squeeze(val_pred)

            Features.append(val_pred)
            set_belong.append(1)
            
            
            
            #val_pred = np.array(self.__predict(x_batch_val))
            y_pred = self.__predict(x_batch_val)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1
            Prediction.append(array)
            
            q += 1
        for x_batch_val, y_batch_val in self.Validation_data:
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], n_batch_val])
            val_pred = self.Extract([x_batch_val], training=False)
            #val_pred = self.Extract([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], [n_batch_val, q_batch_val]], training=False)
            val_pred = np.squeeze(val_pred)

            Features.append(val_pred)
            set_belong.append(0)
            
            
            
            #val_pred = np.array(self.__predict(x_batch_val))
            y_pred = self.__predict(x_batch_val)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1
            Prediction.append(array)
            
            q += 1
            
        Features = np.array(Features)
        Prediction = np.array(Prediction)
        self.df_combined["belong_set"] = set_belong
        #np.save("{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Features)
        #np.save("{}_{}_{}_{}_names.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Name)
        #clusterable_embedding = umap.UMAP()
        #cluster = clusterable_embedding.fit_transform(Features)
        pca = PCA(n_components=3)
        cluster2 = pca.fit_transform(Features)
        if self.args.features:

            self.df_combined["X_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 0]
            self.df_combined["Y_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 1]
            self.df_combined["Z_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 2]
        else:
            

            self.df_combined["X_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 0]
            self.df_combined["Y_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 1]
            self.df_combined["Z_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 2]

        #p = os.path.join(self.args.post_analysis_folder, "{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.img_size, self.args.analysis_type, self.args.backbone))
        #np.save(p, Features)
        count = 0
        number_point = []
        for item in self.args.prediction_classes:
            item2 = item + "_pred"
            self.df_combined[item2] = Prediction[:,count]
            count += 1
        print(self.df_combined.shape)
        for row in range(0, self.df_combined.shape[0]):
            x,y,z = self.df_combined["X_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row], self.df_combined["Y_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row], self.df_combined["Z_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row]
            df2 = self.df_combined.drop(self.df_combined.index[row])
            pred = np.array([self.df_combined[self.args.prediction_classes[0]+ "_pred"].iloc[row], self.df_combined[self.args.prediction_classes[1] +"_pred"].iloc[row]])
            NN_score = self.NN(df2,x,y, z, pred)
            number_point.append(NN_score)
        
        self.df_combined["NN_stage1"] = number_point
        self.df_combined.to_excel("PCA_POSITIONS_stage1.xlsx")
        
        
        return self.df_combined, "NN_stage1"
    def Extract_features_threasholding(self, epoch=None):
        print("Running feature Extraction")
        Features = []
        Prediction = []
        set_belong = []
        if self.args.connection_type == "Average" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[141].output)
        elif self.args.connection_type == "Baseline" and self.args.backbone == "U2":
            self.Extract = Model(
                self.model.inputs, self.model.layers[96].output)
        else:
            self.Extract = Model(
                self.model.inputs, self.model.layers[-2].output)
        if self.args.features == False:
            self.df = self.Train_data.get_dataframe()
            self.df_val = self.Validation_data.get_dataframe()
        q = 0
        self.df_combined = pd.concat([self.df, self.df_val], axis=0)
        
        # for x_batch_val, y_batch_val, z_batch_val, u_batch_val, w_batch_val, n_batch_val, q_batch_val in self.Validation_data:
        for x_batch_val, y_batch_val in self.Train_data:
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], n_batch_val])
            val_pred = self.Extract([x_batch_val], training=False)
            #val_pred = self.Extract([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], [n_batch_val, q_batch_val]], training=False)
            val_pred = np.squeeze(val_pred)

            Features.append(val_pred)
            set_belong.append(1)
            
            
            
            #val_pred = np.array(self.__predict(x_batch_val))
            y_pred = self.__predict(x_batch_val)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1
            Prediction.append(array)
            
            q += 1
        for x_batch_val, y_batch_val in self.Validation_data:
            #y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], n_batch_val])
            val_pred = self.Extract([x_batch_val], training=False)
            #val_pred = self.Extract([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val], [n_batch_val, q_batch_val]], training=False)
            val_pred = np.squeeze(val_pred)

            Features.append(val_pred)
            set_belong.append(0)
            
            
            
            #val_pred = np.array(self.__predict(x_batch_val))
            y_pred = self.__predict(x_batch_val)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1
            Prediction.append(array)
            
            q += 1
            
        Features = np.array(Features)
        Prediction = np.array(Prediction)
        self.df_combined["belong_set"] = set_belong
        #np.save("{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Features)
        #np.save("{}_{}_{}_{}_names.npy".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size),Name)
        #clusterable_embedding = umap.UMAP()
        #cluster = clusterable_embedding.fit_transform(Features)
        pca = PCA(n_components=2)
        cluster2 = pca.fit_transform(Features)
        if self.args.features:

            self.df_combined["X_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 0]
            self.df_combined["Y_{}_PCA_POSITIONS_epoch".format(
                self.args.connection_type)] = cluster2[:, 1]
            #self.df_combined["Z_{}_PCA_POSITIONS_epoch".format(
                #self.args.connection_type)] = cluster2[:, 2]
        else:
            

            self.df_combined["X_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 0]
            self.df_combined["Y_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 1]
            #self.df_combined["Z_{}_PCA_POSITIONS".format(self.args.connection_type)] = cluster2[:, 2]

        #p = os.path.join(self.args.post_analysis_folder, "{}_{}_{}_{}.npy".format(self.args.connection_type, self.args.img_size, self.args.analysis_type, self.args.backbone))
        #np.save(p, Features)
        count = 0
        number_point = []
        for item in self.args.prediction_classes:
            item2 = item + "_pred"
            self.df_combined[item2] = Prediction[:,count]
            count += 1
        Train = self.df_combined[(self.df_combined["belong_set"] == 1)]
        Test = self.df_combined[(self.df_combined["belong_set"] == 0)]
        xh_centre, yh_centre, xm_centre, ym_centre = self.Find_centre_point(Train)
        
        
        count = 0 
        NN_score = self.N_centre(Test,xh_centre, yh_centre,  xm_centre, ym_centre)
        for item in self.args.prediction_classes:
            item2 = item + "_thresh"
            Test[item2] = NN_score[count]
            count += 1
        Test.to_excel("PCA_threashold_stage1.xlsx")
        Val_PRED = np.array(Test.loc[:, [self.args.prediction_classes[0]+ "_thresh", self.args.prediction_classes[1]+ "_thresh"]])
        Val_GRON = np.array(Test.loc[:, [self.args.prediction_classes[0], self.args.prediction_classes[1]]])
        ACC = accuracy_score(Val_GRON, Val_PRED)
        print(ACC)
        return Test
    
    def NN(self,data,x,y,z, pred):
        ED_score = []
        count = 0
        for row in range(0, data.shape[0]):
            xp,yp, zp = data["X_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row], data["Y_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row], data["Z_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row]
            ED = self.euclidean_distance(x,y,z, xp,yp, zp)
            ED_score.append(ED)
        
        for ii in range(0, self.args.nn):
            miminmum = min(ED_score)
            pos = ED_score.index(miminmum)
            check_class = np.array([data[self.args.prediction_classes[0]+ "_pred"].iloc[pos], data[self.args.prediction_classes[1] +"_pred"].iloc[pos]])

            if (pred == check_class).all():
                count += 1
                ED_score[pos] = max(ED_score)
            else:
                ED_score[pos] = max(ED_score)
        
        return count
    
    def N_centre(self,data,xh_centre, yh_centre,  xm_centre, ym_centre):
        High_score = []
        Med_score = []
        count = 0
        ground = np.array(data.loc[:, [self.args.prediction_classes[0], self.args.prediction_classes[1]]])
        for row in range(0, data.shape[0]):
            xp,yp = data["X_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row], data["Y_{}_PCA_POSITIONS".format(self.args.connection_type)].iloc[row]
            High = self.euclidean_distance(xh_centre, yh_centre,  xp,yp)
            Med = self.euclidean_distance(xm_centre, ym_centre,  xp,yp)
            g = ground[row,:]
            if High > Med:
                
                High_score.append(1)
                Med_score.append(0)
                print("High:{}, Med:{}, Pred:{}".format(1,0,g))
            elif High < Med:

                High_score.append(0)
                Med_score.append(1)
                print("High:{}, Med:{}, Pred:{}".format(0,1,g))
            
        
        
        return [High_score, Med_score]
    
    def Find_centre_point(self,data):
        High = data[(data[self.args.prediction_classes[0]] == 1)]
        Med = data[(data[self.args.prediction_classes[0]] == 0)]
        xh = list(High["X_{}_PCA_POSITIONS".format(self.args.connection_type)])
        yh = list(High["Y_{}_PCA_POSITIONS".format(self.args.connection_type)])
        #zh = list(High["Z_{}_PCA_POSITIONS".format(self.args.connection_type)])
        xh_centre = round(sum(xh)/len(xh), 2)
        yh_centre = round(sum(yh)/len(yh), 2)
        #zh_centre = round(sum(zh)/len(zh), 2)
        
        xm = list(Med["X_{}_PCA_POSITIONS".format(self.args.connection_type)])
        ym = list(Med["Y_{}_PCA_POSITIONS".format(self.args.connection_type)])
        #zm = list(Med["Z_{}_PCA_POSITIONS".format(self.args.connection_type)])
        xm_centre = round(sum(xm)/len(xm), 2)
        ym_centre = round(sum(ym)/len(ym), 2)
        #zm_centre = round(sum(zm)/len(zm), 2)
        
        return xh_centre, yh_centre, xm_centre, ym_centre
    
    def euclidean_distance(self, xg,yg, xp,yp):
        distance = 0.0
        distance += (xg - xp)**2
        distance += (yg - yp)**2
        #distance += (zg - zp)**2
        return sqrt(distance)

    def create_confusion(self, Val_PRED, Val_GRON):
        conf = np.zeros((Val_PRED.shape[1], Val_PRED.shape[1]))

        for row in range(0, Val_PRED.shape[0]):
            g = int(np.argmax(Val_GRON[row]))
            p = int(np.argmax(Val_PRED[row]))
            conf[g, p] += 1

        return conf
    def convert_arrays_2to1(self, array):
        collection = []
        for row in range(0, array.shape[0]):
            collection.append(np.argmax(array[row,:]))
        collection = np.array(collection)
        return collection
    
    def predict_model_val(self):
        Val_PRED = []
        Val_GRON = []
        Val_raw = []
        for x_batch_val, y_batch_val in self.Validation_data:
        #for x_batch_val, y_batch_val,  in self.Validation_data:

            y_pred = self.__predict([[x_batch_val]])
            #y_pred = self.__predict([x_batch_val])
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            Val_raw.append(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1

            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))

            prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)

        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        Val_raw = np.squeeze(np.array(Val_raw))

        if self.args.num_output_classes > 2:
            ACC2 = keras.metrics.CategoricalAccuracy()
        else:
            ACC2 = keras.metrics.BinaryAccuracy()

        ACC2.update_state(Val_GRON, Val_PRED)
        ACC_SCORE = ACC2.result().numpy()
        g, t = self.convert_arrays_2to1(Val_GRON), self.convert_arrays_2to1(Val_PRED)
        rscore = rand_score(g,t)
        mscore = mutual_info_score(g,t)
        admscore = adjusted_mutual_info_score(g,t)
        rocscore = roc_auc_score(Val_GRON, Val_raw, multi_class='ovr')
        print(rscore, mscore, admscore,rocscore)
        return ACC_SCORE, rscore, mscore, admscore, rocscore
    
        
        
    def predict_model_train(self):
        Val_PRED = []
        Val_GRON = []
        Val_raw = []
        for x_batch_val, y_batch_val, in self.Train_data:
        #for x_batch_val, y_batch_val,  in self.Validation_data:

            y_pred = self.__predict([[x_batch_val]])
            
            #y_pred = self.__predict([x_batch_val])
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            Val_raw.append(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1

            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))

            prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)

        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        Val_raw = np.squeeze(np.array(Val_raw))
        if self.args.num_output_classes > 2:
            ACC2 = keras.metrics.CategoricalAccuracy()
        else:
            ACC2 = keras.metrics.BinaryAccuracy()
        self.plot_confusion(Val_PRED, Val_GRON)
        ACC2.update_state(Val_GRON, Val_PRED)
        ACC_SCORE = ACC2.result().numpy()
        g, t = self.convert_arrays_2to1(Val_GRON), self.convert_arrays_2to1(Val_PRED)
        rscore = rand_score(g,t)
        mscore = mutual_info_score(g,t)
        admscore = adjusted_mutual_info_score(g,t)
        rocscore = roc_auc_score(Val_GRON, Val_raw, multi_class='ovr')
        print(rscore, mscore, admscore,rocscore)

        return ACC_SCORE, rscore, mscore, admscore, rocscore
    

    def predict_model(self):
        Val_p = []
        Val_PRED = []
        Val_GRON = []
        save_place = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold))
        if self.args.features == False:
            self.df = self.Validation_data.get_dataframe()
        if os.path.exists(save_place) == False:
            os.mkdir(save_place)
        for x_batch_val, y_batch_val in self.Validation_data:
        #for x_batch_val, y_batch_val,  in self.Validation_data:

            y_pred = self.__predict([[x_batch_val]])
            #y_pred = self.__predict([x_batch_val])
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            prediction_array = np.zeros((y_batch_val.shape))
            for ii in range(0, y_pred.shape[0]):
                Pos = np.argmax(y_pred[ii,:])
                prediction_array[ii,Pos] = 1
                
                
            
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            
            
            #prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)

        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        

        print(Val_PRED.shape, Val_GRON.shape)
        CON_MATRIX = skm.multilabel_confusion_matrix(Val_GRON, Val_PRED)
        print(CON_MATRIX)
        self.plot_confusion(Val_PRED, Val_GRON)
        Class_report = skm.classification_report(Val_GRON, Val_PRED, target_names=self.args.prediction_classes)
        print(Class_report)
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}.txt".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")
        con = self.create_confusion(Val_PRED, Val_GRON)
        print(con)
        file1.writelines(Class_report)
        file1.close()
        ACC = accuracy_score(Val_GRON, Val_PRED)
        print(ACC)
        if self.args.num_output_classes > 2:
            ACC2 = keras.metrics.CategoricalAccuracy()
        else:
            ACC2 = keras.metrics.BinaryAccuracy()

        ACC2.update_state(Val_GRON, Val_PRED)
        ACC_SCORE = ACC2.result().numpy()
        print(ACC_SCORE)
        
        m = self.val_acc_metric.result()
        print(m)
        obj = [self.args.prediction_classes[0], self.args.prediction_classes[1]]
        count= 0
        for item in obj:
            item2 = item + "_pred"
            self.df[item2] = Val_PRED[:,count]
            count += 1
        p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}_{}.xlsx".format(self.args.connection_type,self.args.test_fold, self.args.repeat, self.args.img_size, self.args.tag))
        self.df.to_excel(p)

        return ACC_SCORE
    
    
    def tensorflow_analysis(self):
        for x_batch_val, y_batch_val, z_batch_val in self.Validation_data:
            y_pred = self.__predict([[x_batch_val, z_batch_val]])

            self.val_acc_metric.update_state(y_batch_val, y_pred)
            

        m = self.val_acc_metric.result()
        
        print(m)
        self.val_acc_metric.reset_states()

        return m
    
    def bb_intersection_over_union(self, boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
    	xA = max(boxA[0], boxB[0])
    	yA = max(boxA[1], boxB[1])
    	xB = min(boxA[2], boxB[2])
    	yB = min(boxA[3], boxB[3])
    	# compute the area of intersection rectangle
    	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    	# compute the area of both the prediction and ground-truth
    	# rectangles
    	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    	# compute the intersection over union by taking the intersection
    	# area and dividing it by the sum of prediction + ground-truth
    	# areas - the interesection area
    	iou = interArea / float(boxAArea + boxBArea - interArea)
    	# return the intersection over union value
    	return iou
    
    def predict_regression(self):

        Val_PRED = []
        Val_GRON = []
        iou_score = []
        for x_batch_val, y_batch_val in self.Validation_data:

            #y_pred = self.__predict([[x_batch_val, z_batch_val]])
            y_pred = self.__predict([x_batch_val])
            #self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            #print(y_pred)
            y_batch_val = y_batch_val*self.args.img_size

            y_batch_val= y_batch_val.astype(int)

            y_pred = y_pred*self.args.img_size
            
            y_pred = y_pred.astype(np.uint8)
            gzmin,gxmin,gzmax,gxmax = y_batch_val[0,:]
            #print(y_batch_val, y_pred)
            pzmin,pxmin,pzmax,pxmax = y_pred[0,:]
            
            
            x_batch_val = np.squeeze(x_batch_val)*255.0
            x_batch_val = x_batch_val.astype(np.uint8)
            #print(x_batch_val.shape)
            #cv2.imwrite("test.png", x_batch_val)
            #x_batch_val = cv2.imread("test.png")
            cv2.rectangle(x_batch_val,(gxmin,gzmin),(gxmax,gzmax),(255,0,0),2)
            cv2.rectangle(x_batch_val,(pxmin,pzmin),(pxmax,pzmax),(0,255,0),2)
            plt.imshow(x_batch_val)
            #print(zmax,zmin,xmax,xmin)
            #print(zmaxp,zminp,xmaxp,xminp)
            #iou = self.bb_intersection_over_union([gzmin,gxmin,gzmax,gxmax], [pzmin,pxmin,pzmax,pxmax])
            
            #iou_score.append(iou)
            #print(x_batch_val.shape)
            #print(iou)
            #break
            #self.val_acc_metric.update_state(y_batch_val, y_pred)
            #print(self.val_acc_metric.result())
            break

        m = self.val_acc_metric.result()
        
        print(m)
        self.val_acc_metric.reset_states()

        iou_value = np.mean(iou_score)
        print(iou_value)
        return iou_value

    def predict_model2(self):

        Val_PRED = []
        Val_GRON = []
        for x_batch_val, y_batch_val, z_batch_val, u_batch_val, w_batch_val in self.Validation_data:

            y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val]])
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1

            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))

            prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)

        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))

        print(Val_PRED.shape, Val_GRON.shape)
        CON_MATRIX = skm.multilabel_confusion_matrix(Val_GRON, Val_PRED)
        print(CON_MATRIX)
        con = self.create_confusion(Val_PRED, Val_GRON)
        print(con)
        #self.plot_confusion(Val_PRED, Val_GRON)
        Class_report = skm.classification_report(
            Val_GRON, Val_PRED, target_names=self.args.prediction_classes)
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "{}_{}_{}_{}.txt".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")

        file1.writelines(Class_report)
        file1.close()
        ACC = accuracy_score(Val_GRON, Val_PRED)
        print(ACC)
        if self.args.num_output_classes > 2:
            ACC2 = keras.metrics.CategoricalAccuracy()
        else:
            ACC2 = keras.metrics.BinaryAccuracy()

        ACC2.update_state(Val_GRON, Val_PRED)
        ACC_SCORE = ACC2.result().numpy()
        print(ACC_SCORE)

        m = self.val_acc_metric.result()
        print(m)

        return ACC

    def predict_model_analysis(self, Val_GRON, Val_PRED):
        CON_MATRIX = skm.multilabel_confusion_matrix(Val_GRON, Val_PRED)
        print(CON_MATRIX)
        self.plot_confusion(Val_PRED, Val_GRON)
        Class_report = skm.classification_report(
            Val_GRON, Val_PRED, target_names=self.args.prediction_classes)
        file_p = os.path.join(self.args.post_analysis_folder, "{}_{}.txt".format(
            self.args.connection_type, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")
        file1.writelines(Class_report)
        file1.close()
        ACC = accuracy_score(Val_GRON, Val_PRED)

        print(ACC)
        return ACC

    def predict_model_fold(self):
        Val_PRED = []
        Val_GRON = []
        NAMES = []
        Name_excel = self.Validation_data.get_dataframe()
        count = 0
        for x_batch_val,  y_batch_val in self.Validation_data:
            val_pred = np.array(self.__predict(x_batch_val))
            POS = np.argmax(val_pred)
            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))
            prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)
            NAMES.append(Name_excel[self.args.image_col].iloc[count])
            count += 1
        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        NAMES = np.array(NAMES)
        #ACC = accuracy_score(Val_GRON, Val_PRED)
        #print(ACC)
        return Val_GRON, Val_PRED, NAMES

    def predict_model_excel(self):
        ACC_LIST = []
        NAMES = []
        INV = []
        DCIS = []
        count = 0
        Name_excel = self.Validation_data.get_dataframe("LesionID")
        for x_batch_val, z_batch_val, y_batch_val in self.Validation_data:
            val_pred = np.array(self.__predict(x_batch_val,  z_batch_val))
            POS = np.argmax(val_pred)
            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))

            prediction_array[:, POS] = 1
            if (prediction_array == y_batch_val).all():
                ACC_LIST.append(1)
            else:
                ACC_LIST.append(0)
            NAMES.append(Name_excel.iloc[count])
            INV.append(y_batch_val[0, 1])
            DCIS.append(y_batch_val[0, 0])

            count += 1
        print(ACC_LIST)
        return NAMES, ACC_LIST, INV, DCIS

    def predict_images_from_Dictory(self, img_list, base):
        data = pd.DataFrame()
        Class = []
        q = 0
        for img in img_list:
            q += 1
            path = os.path.join(base, img)
            im = cv2.imread(path, 0)
            image_r = cv2.resize(im, (self.args.img_size, self.args.img_size),
                                 interpolation=cv2.INTER_CUBIC)  # set to 256x256
            image_r = image_r/np.max(image_r)
            image_r = np.expand_dims(image_r, axis=-1)
            image_r = np.expand_dims(image_r, axis=0)
            val_pred = np.array(self.__predict(image_r))
            POS = np.argmax(val_pred)
            Class.append(POS)
            print(q)
        data["Names"] = img_list
        data["Class"] = Class
        data.to_excel("Dataset_refining2.xlsx")
        return q

    def create_training_logs(self, training, validation, Final_epochs):
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "Training_logs_{}_{}_{}_{}.txt".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")
        report_list = []
        for epoch in range(0, Final_epochs):
            report_list.append("Training_loss: {}, Validation_loss:{} \n".format(
                training[epoch], validation[epoch]))
        report = "".join(report_list)
        file1.writelines(report)
        file1.close()

        return None

    def Train_model(self):
        count = 0
        Training_loss = []
        Validation_loss = []
        if self.args.features:
            self.df = self.test_data.get_dataframe()
        epo = []
        self.df_train = self.Train_data.get_dataframe()
        metrics_names = ['train_loss']
        self.class_weights = np.zeros((len(self.args.prediction_classes)))
        for option in range(0, len(self.args.prediction_classes)):
            self.class_weights[option] = sum(list(self.df_train[self.args.prediction_classes[option]]))/self.df_train.shape[0]
        print("Classwieghts:", self.class_weights)
        for epoch in range(self.args.max_epochs):
            epo.append(epoch)
            validation_acc = []
            self.Schedule.on_epoch_begin(epoch, self.model.optimizer)
            start_time = time.time()
            pb_i = Progbar(self.Train_data.n, stateful_metrics=metrics_names)
            # Iterate over the batches of the dataset.
            train_loss_value = []
            
            
            if self.args.num_input_objects == 2 and self.args.num_output_objects > 1:
                for step, (x_batch_train, z_batch_train, y_batch_train, w_batch_train) in enumerate(self.Train_data):
                    
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train], [y_batch_train, w_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train], [y_batch_train, w_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
            
            
            
            
            elif self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                for step, (x_batch_train, y_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train], [y_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train], [y_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
            
            elif self.args.num_input_objects == 4 and self.args.num_output_objects == 1:
                for step, (x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, u_batch_train, w_batch_train], [y_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, u_batch_train, w_batch_train], [y_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
                    
                    
                    
            elif self.args.num_input_objects == 5 and self.args.num_output_objects == 1:
                for step, (x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, u_batch_train, w_batch_train, n_batch_train], [y_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, u_batch_train, w_batch_train, n_batch_train], [y_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
                    
                    
            elif self.args.num_input_objects == 6 and self.args.num_output_objects == 1:
                for step, (x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train], [y_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train], [y_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
                    

                    
            elif self.args.num_input_objects == 2 and self.args.num_output_objects == 1:
                for step, (x_batch_train, y_batch_train, z_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train], [y_batch_train])
                    else:
                        
                        loss_value, y_pred = self.train_step([x_batch_train, z_batch_train], [y_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
                    
                    
            elif self.args.num_input_objects == 1 and self.args.num_output_objects > 1:
                for step, (x_batch_train, y_batch_train, w_batch_train) in enumerate(self.Train_data):
                    if self.args.model_type == 'Segmentation':
                        loss_value, y_pred = self.train_step([x_batch_train], [y_batch_train, w_batch_train])
                    else:
                        loss_value, y_pred = self.train_step([x_batch_train], [y_batch_train, w_batch_train])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)

            # Display metrics at the end of each epoch.
            Training_loss.append(np.mean(train_loss_value))
            self.Train_acc[epoch] = self.train_acc_metric.result()
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            Validation_loss_value = []
            if self.args.num_input_objects == 2 and self.args.num_output_objects > 1:
                for x_batch_val, z_batch_val, y_batch_val, w_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val, z_batch_val], [y_batch_val, w_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                for x_batch_val, y_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val], [y_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 4 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 5 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)
                
                
            elif self.args.num_input_objects == 6 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)

            elif self.args.num_input_objects == 1 and self.args.num_output_objects > 1:
                for x_batch_val, y_batch_val, w_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val], [y_batch_val, w_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
            elif self.args.num_input_objects == 2 and self.args.num_output_objects == 1:
                for x_batch_val, y_batch_val, z_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val, z_batch_val],[y_batch_val])
                    Validation_loss_value.append(loss_value)


            Validation_loss.append(np.mean(Validation_loss_value))
            self.Val_acc[epoch] = self.val_acc_metric.result()
                
            if self.args.features:
                create_features = self.Extract_features(epoch=epoch)
            if self.args.lr_schedule == "Cosine":
                self.Schedule.on_epoch_end(epoch, self.model.optimizer,Training_loss[-1])
            else:
                self.Schedule.on_epoch_end(epoch, Validation_loss[-1])
            print("Val loss:{} Val accuracy:{} ".format(Validation_loss[-1], self.Val_acc[epoch]))
            if np.argmax(self.Val_acc) == epoch or epoch == 0:
                self.Save_model_weight()
                print("Weights Saved")
                count = 0

                cut_off = 5
                print("End Count: {} Cut off at: {}".format(count, cut_off))
            else:
                print("End Count: {} Cut off at: {}".format(count, cut_off))
                if epoch > self.args.cooldown:
                    count += 1

            self.Train_data.on_epoch_end()
            self.val_acc_metric.reset_states()
            if count == cut_off:
                if self.args.early_finish == "True":
                    print("breaking loop")
                    break
        #self.create_training_logs(Training_loss, Validation_loss, max(epo))
        
    def predict_model_batch(self):
        if self.args.features:
            self.df = self.test_data.get_dataframe()
        epo = []
        self.df_train = self.Train_data.get_dataframe()
        metrics_names = ['train_loss']
        self.class_weights = np.zeros((len(self.args.prediction_classes)))
        for option in range(0, len(self.args.prediction_classes)):
            self.class_weights[option] = sum(list(self.df_train[self.args.prediction_classes[option]]))/self.df_train.shape[0]
        print("Classwieghts:", self.class_weights)
        p = 0
        if p ==0:
            Validation_loss_value = []
            if self.args.num_input_objects == 2 and self.args.num_output_objects > 1:
                for x_batch_val, z_batch_val, y_batch_val, w_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val, z_batch_val], [y_batch_val, w_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                for x_batch_val, y_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val], [y_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 4 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)
                
                
                
            elif self.args.num_input_objects == 5 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)
                
                
            elif self.args.num_input_objects == 6 and self.args.num_output_objects == 1:
                for x_batch_train, y_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train in self.Validation_data:
                    loss_value = self.test_step([x_batch_train, z_batch_train, w_batch_train, u_batch_train, n_batch_train, p_batch_train], [y_batch_train])
                    Validation_loss_value.append(loss_value)

            elif self.args.num_input_objects == 1 and self.args.num_output_objects > 1:
                for x_batch_val, y_batch_val, w_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val], [y_batch_val, w_batch_val])
                    Validation_loss_value.append(loss_value)
                
                
            elif self.args.num_input_objects == 2 and self.args.num_output_objects == 1:
                for x_batch_val, y_batch_val, z_batch_val in self.Validation_data:
                    loss_value = self.test_step([x_batch_val, z_batch_val],[y_batch_val])
                    Validation_loss_value.append(loss_value)


            
            m =  self.val_acc_metric.result()
                
        return m

    def heatmap_process(self, heatmap):
        normdataset = np.maximum(heatmap, 0)
        normdataset = normdataset / np.max(normdataset)
        return (normdataset)

    def watch_layer(self,layer, tape):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Store the result of `layer.call` internally.
                layer.result = func(*args, **kwargs)
                # From this point onwards, watch this tensor.
                tape.watch(layer.result)
                print("the wrapper is being used")
                # Return the result to continue with the forward pass.
                return layer.result
            return wrapper
        layer.call = decorator(layer.call)
        return layer
    

    def run_hi_res_cam(self):
        
        self.model.layers[-1].activation = tf.keras.activations.linear
        model_layer_list = list(self.model.layers)
        for lay in range(0, len(model_layer_list)):
            if "Activation" in str(model_layer_list[lay]):
                top_conv = self.model.layers[lay]
                break

        IMAGE_SAVE = os.path.join(
            self.args.post_analysis_folder, "Attention_Masks")
        if os.path.exists(IMAGE_SAVE) == False:
            os.mkdir(IMAGE_SAVE)
        if self.args.features == False:
            self.df = self.Validation_data.get_dataframe()
        q = 0
        for x_batch_val, y_batch_val in self.Validation_data:
            #tf.compat.v1.reset_default_graph()
            IMG_NAME = str(self.df[self.args.image_col].iloc[q]) + self.args.ext_type
            with tf.GradientTape() as gtape:
                # Make the `last_conv_layer` watchable
                self.watch_layer(top_conv, gtape)
                preds = self.model(x_batch_val)
                pred_index = tf.argmax(preds[0])
                class_output = preds[:, pred_index]
                #print(class_output)
            
            grads = gtape.gradient(class_output, top_conv.result)
            grads = grads[0]
            
            
            iterate = K.function([self.model.input], [grads, top_conv.result[0]])
            grads_x, top_conv_x = iterate([x_batch_val])

            for ii in range(top_conv_x.shape[2]):
                top_conv_x[:, :, ii] *= grads_x[:, :, ii]
                heatmap = top_conv_x.sum(axis=-1)
            heatmap_pro = self.heatmap_process(heatmap)
            #cv2.imwrite("test1.png", heatmap_pro)
            #cv2.imwrite("test3.png", X_test[0,:,:,:]*255)
            jet = cm.get_cmap("jet")
            heatmap = np.uint8(255 * heatmap_pro)
            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize(
                (x_batch_val.shape[1], x_batch_val.shape[2]))
            jet_heatmap = image.img_to_array(jet_heatmap)

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * 0.5 + x_batch_val[0, :, :]
            superimposed_img = np.uint8(255.0 * (superimposed_img - np.min(superimposed_img)) / (
                np.max(superimposed_img) - np.min(superimposed_img)))
            superimposed_img = cv2.cvtColor(
                superimposed_img, cv2.COLOR_BGR2GRAY)
            OUTPUT_PATH = os.path.join(IMAGE_SAVE, IMG_NAME)
            #plt.imshow(superimposed_img)
            superimposed_img = cv2.resize(superimposed_img, (512, 512),
                                 interpolation=cv2.INTER_CUBIC)  # set to 256x256
            cv2.imwrite(OUTPUT_PATH, superimposed_img)
            q += 1
        return IMG_NAME
    
    def prediction_fixtures(self):
        IMAGE_SAVE = os.path.join(self.args.post_analysis_folder, "{}_{}_{}.xlsx".format(self.args.val_fold, self.args.backbone, self.args.tag))
        Val_PRED = []
        Val_GRON = []
        if self.args.features == False:
            self.df = self.Validation_data.get_dataframe()
        for x_batch_val, y_batch_val in self.Validation_data:

            y_pred = self.__predict([x_batch_val])
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            y_pred = np.array(y_pred)
            POS = np.argmax(y_pred)
            array = np.zeros((len(self.args.prediction_classes)))
            array[POS] = 1

            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))

            prediction_array[:, POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)

        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))

        self.df["{}_fold_{}".format(self.args.val_fold, self.args.prediction_classes[0])] = Val_PRED[:,0]
        self.df["{}_fold_{}".format(self.args.val_fold, self.args.prediction_classes[1])] = Val_PRED[:,1]
        
        self.df.to_excel(IMAGE_SAVE)

        return None
    
    def run_hi_res_cam2(self, pred_index):
        
        self.model.layers[-1].activation = None
        model_layer_list = list(self.model.layers)
        for lay in range(0, len(model_layer_list)):
            if "Activation" in str(model_layer_list[lay]):
                top_conv = self.model.layers[lay]
                break

        step_SAVE = os.path.join(
            self.args.post_analysis_folder, str(self.args.test_fold))
        if os.path.exists(step_SAVE) == False:
            os.mkdir(step_SAVE)
        IMAGE_SAVE = os.path.join(
            step_SAVE, "Attention_maps")
        if os.path.exists(IMAGE_SAVE) == False:
            os.mkdir(IMAGE_SAVE)
        if self.args.features == False:
            self.df = self.Validation_data.get_dataframe()
        q = 0
        for x_batch_val, y_batch_val in self.Validation_data:
            # First, we create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions
            IMG_NAME = str(self.df[self.args.image_col].iloc[q]) + self.args.ext_type
            grad_model = tf.keras.models.Model([self.model.inputs], [top_conv.output, self.model.output])
            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(x_batch_val)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
            x = np.squeeze(x_batch_val)
            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            OUTPUT_PATH = os.path.join(IMAGE_SAVE, IMG_NAME)
            
            #normdataset = np.maximum(heatmap, 0)
            #heatmap = normdataset / np.max(normdataset)
            #plt.imshow(superimposed_img)
            #img = (heatmap.numpy())*255
            heatmap = np.uint8(255 * heatmap.numpy())
            #cv2.imwrite(OUTPUT_PATH, heatmap)
            #plt.imsave(OUTPUT_PATH, heatmap,cmap ='gray')
            #heatmap = np.uint8(255 * heatmap)
            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((512,512))
            jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
            #superimposed_img = jet_heatmap * 0.4 + x*255
            #superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

            cv2.imwrite(OUTPUT_PATH, heatmap)
            #cv2.imwrite(OUTPUT_PATH, heatmap)
            #superimposed_img.save(OUTPUT_PATH)
            print("you are on item {}".format(q))
            q += 1
        return IMG_NAME
    def Train_model_regression(self):
        count = 0
        Training_loss = []
        Validation_loss = []

        epo = []
        metrics_names = ['train_loss']
        for epoch in range(self.args.max_epochs):
            print("epoch:{}/{}".format(epoch, self.args.max_epochs))
            epo.append(epoch)
            validation_acc = []
            if self.args.lr_schedule == "cosine":
                self.Schedule.on_epoch_begin(epoch, self.model.optimizer)
            start_time = time.time()
            pb_i = Progbar(self.Train_data.n, stateful_metrics=metrics_names)
            # Iterate over the batches of the dataset.
            train_loss_value = []

            for step, (x_batch_train, y_batch_train) in enumerate(self.Train_data):
                sample_weights = np.zeros(self.args.batch_size)

                loss_value, y_pred = self.train_step_regression(x_batch_train, y_batch_train,  sample_weights)
                values = [('train_loss', loss_value)]
                pb_i.add(self.args.batch_size, values=values)
            # Display metrics at the end of each epoch.
            Training_loss.append(np.mean(train_loss_value))
            self.Train_acc[epoch] = self.train_acc_metric.result()
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            Validation_loss_value = []

            for x_batch_val, y_batch_val in self.Validation_data:
                loss_value = self.test_step_regression(x_batch_val, y_batch_val)
                Validation_loss_value.append(loss_value)
            Validation_loss.append(np.mean(Validation_loss_value))
            self.Val_acc[epoch] = self.val_acc_metric.result()

            if self.args.lr_schedule == "cosine":
                self.Schedule.on_epoch_end(epoch, self.model.optimizer)
            elif self.args.lr_schedule == "exp_decay":
                self.Schedule.on_epoch_end(self.args, epoch, self.Val_acc[epoch])
            print(self.Val_acc[epoch])
            if np.argmax(self.Val_acc) == epoch or epoch == 0:
                self.Save_model_weight()
                print("Weights Saved")
                count = 0

                cut_off = 5
                print("End Count: {} Cut off at: {}".format(count, cut_off))
            else:
                print("End Count: {} Cut off at: {}".format(count, cut_off))
                if epoch > self.args.cooldown:
                    count += 1

            self.Train_data.on_epoch_end(epoch)
            self.val_acc_metric.reset_states()
            if count == cut_off:
                if self.args.early_finish == "True":
                    print("breaking loop")
                    break
        #self.create_training_logs(Training_loss, Validation_loss, max(epo))
        self.Save_model_weight()


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, y_true, y_pred):
        if self.args.num_output_objects > 1:
            left_cal = self.dice(y_true[0], y_pred[0])
            left_loss = 1 - left_cal
            right_cal = self.dice(y_true[1], y_pred[1])
            right_loss = 1 - right_cal
            diff = tf.math.abs(left_cal - right_cal)
            average_loss = left_loss + right_loss
            return average_loss
        elif self.args.num_output_objects == 1:
            cal = 1 - self.dice(y_true, y_pred)
            return cal

    def dice(self, y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / \
            (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice
class IOU_Loss_Regression(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
    def call(self, y_true, y_pred):
        if self.args.num_output_objects > 1:
            left_cal = self.dice(y_true[0], y_pred[0])
            left_loss = 1 - left_cal
            right_cal = self.dice(y_true[1], y_pred[1])
            right_loss = 1 - right_cal
            diff = tf.math.abs(left_cal - right_cal)
            average_loss = left_loss + right_loss
            return average_loss
        elif self.args.num_output_objects == 1:
            values = []
            for row in range(y_true.shape[0]):
                values.append(1 - self.bb_intersection_over_union(y_true[row,:], y_pred[row,:]))
            mean = np.mean(values)
            return mean

    def bb_intersection_over_union(self, boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
    	xA = max(boxA[0], boxB[0])
    	yA = max(boxA[1], boxB[1])
    	xB = min(boxA[2], boxB[2])
    	yB = min(boxA[3], boxB[3])
    	# compute the area of intersection rectangle
    	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    	# compute the area of both the prediction and ground-truth
    	# rectangles
    	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    	# compute the intersection over union by taking the intersection
    	# area and dividing it by the sum of prediction + ground-truth
    	# areas - the interesection area
    	iou = interArea / float(boxAArea + boxBArea - interArea)
    	# return the intersection over union value
    	return iou
        
# determine the (x, y)-coordinates of the intersection rectangle


class DiceLoss_Multiclass(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, y_true, y_pred):
        if self.args.num_output_objects > 1:
            dice1 = 0
            for index in range(y_true[0].shape[-1]):
                dice1 += self.dice(y_true[0][:, :, :, index],
                                   y_pred[0][:, :, :, index])
            dice2 = 0
            for index in range(y_true[0].shape[-1]):
                dice2 += self.dice(y_true[1][:, :, :, index],
                                   y_pred[1][:, :, :, index])
            left_cal = dice1/y_true[0].shape[-1]  # taking average
            left_loss = 1 - left_cal
            right_cal = dice2/y_true[0].shape[-1]
            right_loss = 1 - right_cal
            average_loss = left_loss + right_loss
            return average_loss
        elif self.args.num_output_objects == 1:
            y_true = self.convert_tensor(y_true)
            y_pred = self.convert_tensor(y_pred)
            cal = 1 - self.dice(y_true, y_pred)
            return cal
    def convert_tensor(self, ground):
        ground_list = []

        ground = ground * self.args.img_size
        ground = np.rint(ground)
        ground = ground.astype(np.int32)
        for zz in range(0, ground.shape[0]):
            xmin,zmin,xmax,zmax = ground[zz, :]
            ground_mask = np.zeros((self.args.img_size, self.args.img_size), dtype=np.float32)
            ground_mask[zmin:zmax, xmin:xmax] = 1
            ground_list.append(ground_mask)
        ground_list = np.array(ground_list)
        ground_list = tf.convert_to_tensor(ground_list, dtype=tf.float32)
        return ground_list

    def dice(self, y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) /(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice


class IOULoss(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.iou_met = tfa.losses.GIoULoss()

    def call(self, y_true, y_pred):
        if self.args.num_output_objects > 1:
            left_cal = self.iou_met(y_true[0], y_pred[0])

            right_cal = self.iou_met(y_true[1], y_pred[1])

            average_loss = left_loss + right_loss
            return average_loss
        elif self.args.num_output_objects == 1:
            cal = self.IOU(y_true, y_pred)
            return cal

    def IOU(self, y_true, y_pred):
        IOU_SCORE = tfa.losses.giou_loss(y_true, y_pred, mode="iou")

        return IOU_SCORE

class IOULoss2(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        if not y_pred.dtype.is_floating:
            y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        iou = tf.squeeze(self.iou_value(y_true, y_pred))
        cal = 1 - iou
        return cal

    def iou_value(self, b1, b2):
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height
    
        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height
    
        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        return iou
    
class DICE_Regression_Loss(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        if not y_pred.dtype.is_floating:
            y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        iou = tf.squeeze(self.iou_value(y_true, y_pred))
        cal = 1 - iou
        return cal

    def iou_value(self, b1, b2):
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height
    
        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = (intersect_width * intersect_height)*2.
        
        union_area = b1_area + b2_area 
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        return iou
    
class Focal_DICE_Regression_Loss(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, y_true, y_pred):
        dice = self.iou_value(y_true, y_pred)
        focal = self.sigmoid_focal_crossentropy(y_true, y_pred)
        cal = (1 - dice) + focal
        return cal

    def iou_value(self, b1, b2):
        b2 = tf.convert_to_tensor(b2)
        if not b2.dtype.is_floating:
            b2 = tf.cast(b2, tf.float32)
        b1 = tf.cast(b1, b2.dtype)
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height
    
        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = (intersect_width * intersect_height)*2.
        
        union_area = b1_area + b2_area 
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        iou = tf.squeeze(iou)
        
        return iou
    def sigmoid_focal_crossentropy(self, y_true, y_pred, alpha = 0.25,gamma = 2.0 , from_logits = False):
    
        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")
    
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
    
        # Get the cross_entropy for each entry
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    
        # If logits are provided then convert the predictions into probabilities
        if from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred
    
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = 1.0
        modulating_factor = 1.0
    
        if alpha:
            alpha = tf.cast(alpha, dtype=y_true.dtype)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    
        if gamma:
            gamma = tf.cast(gamma, dtype=y_true.dtype)
            modulating_factor = tf.pow((1.0 - p_t), gamma)
    
        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


class RegressionLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.bb = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO)
        self.bb2 = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO)
        self.Pred = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    def call(self, y_true, y_pred):

        left_cal = self.bb(y_true[0], y_pred[0])
        right_cal = self.bb2(y_true[1], y_pred[1])
        average_loss = left_cal + right_cal
        return average_loss

class Two_Way_IOU(tf.keras.metrics.Metric):

    def __init__(self, name='iou_metric', **kwargs):
        super(Two_Way_IOU, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        if not y_pred.dtype.is_floating:
            y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        iou = tf.squeeze(self.iou_value(y_true, y_pred))
        self.true_positives.assign_add(tf.reduce_mean(iou))
        
    def iou_value(self, b1, b2):
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height
    
        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height
    
        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        return iou

    def result(self):
        return self.true_positives
    def reset_states(self):
        self.true_positives.assign(0.)



class Dice(keras.metrics.Metric):
    def __init__(self):
        super().__init__()

    def dice(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def update_state(self, y_true, y_pred, sample_weight=None):
        left_cal = self.dice(y_true[0], y_pred[0])
        right_cal = self.dice(y_true[1], y_pred[1])
        average = left_cal + right_cal
        return average
        self.tp.assign_add(tf.reduce_sum(tf.cast(true_p, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(false_p, self.dtype)))

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)

    def result(self):
        return self.tp - self.fp
