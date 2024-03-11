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
from sklearn.metrics import accuracy_score
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
from CustomDataGen import CustomDataGen
#import sys

np_config.enable_numpy_behavior()

formatter = logging.Formatter(
    '%(asctime)s - (%(name)s) %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


class CustomModel_Multi:
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
    

    def define_optimizer(self):
        if self.args.Optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.args.Starting_lr)
        elif self.args.Optimizer == "SGD":
            return tf.keras.optimizers.experimental.SGD(learning_rate=self.args.Starting_lr)
    def compute_loss(self, labels, predictions, model_losses):
        per_example_loss = self.loss_fn(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss)
        if model_losses:
          loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def loss_function(self):
        if self.args.model_type == 'Classification':
            if self.args.num_output_classes == 2:
                if self.args.loss_type == "cross":
                    print("loss is good")
                    return keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=self.args.logits)
                    
                elif self.args.loss_type == "mse":
                    return tf.keras.losses.MeanSquaredError()
                elif self.args.loss_type == "focal":
                    return tf.keras.losses.BinaryFocalCrossentropy()
                elif self.args.loss_type == "sparse":
                    return tf.nn.sigmoid_cross_entropy_with_logits()
                elif self.args.loss_type == "hinge":
                    return tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.NONE)

                
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
            return CustomSchedule(args = self.args, patience=2,factor=0.5,verbose=1, optim_lr=self.optimizer.learning_rate, reduce_lin=False, mode="min")
    
    def Prep_training(self, Train_data, Validation_data, test_generator=None):
        c_count = Train_data.Class_Count()
        self.num_batches = Train_data.n
        self.df_train = Train_data.get_dataframe()
        if self.args.multi_process:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
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
        else:
            
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
        
        if self.args.num_input_objects == 2:
            self.Train_data = tf.data.Dataset.from_generator(Train_data, output_types=(tf.float64, tf.int64, tf.float64))
            self.Validation_data = tf.data.Dataset.from_generator(Validation_data, output_types=(tf.float64, tf.int64, tf.float64))
        else:
            self.Train_data = tf.data.Dataset.from_generator(Train_data, output_types=(tf.float64, tf.int64))
            self.Validation_data = tf.data.Dataset.from_generator(Validation_data, output_types=(tf.float64, tf.int64))
        
        self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.Train_data)
        self.val_dist_dataset = self.strategy.experimental_distribute_dataset(self.Validation_data)
        if self.args.features:
            self.test_data = test_generator

        self.Train_acc = np.zeros(
            shape=(self.args.max_epochs,), dtype=np.float32)
        self.Val_acc = np.zeros(
            shape=(self.args.max_epochs,), dtype=np.float32)
        if self.args.multi_process:
            with self.strategy.scope():
                self.model = self.Create_Model_From_args()
                if self.args.load_pretain == True:
                    self.Load_model_weight()
                self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.train_acc_metric)
                #if self.args.mode == "train_finetune":
                    #for layer in self.model.layers[:-5]:
                        #layer.trainable = False

        else:
            self.model = self.Create_Model_From_args()
            if self.args.load_pretain == True:
                self.Load_model_weight()
            self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.train_acc_metric)
        
        self.Class_weights = c_count
        fac_c = 1 * np.sum(c_count)
        for ind in range(len(c_count)):
            score = math.log(fac_c/float(c_count[ind]))
            self.Class_weights[ind] = score if score > 1.0 else 1.0
            
            
    @tf.function
    def distributed_train_step(self, dataset_inputs):
      per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs,))
      return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
       per_replica_losses = self.strategy.run(self.test_step, args=(dataset_inputs,))
       return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                              axis=None)
   
    @tf.function
    def distributed_train_four(self, dataset_inputs):
      per_replica_losses = self.strategy.run(self.train_step_four, args=(dataset_inputs,))
      return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    @tf.function
    def distributed_test_four(self, dataset_inputs):
       per_replica_losses = self.strategy.run(self.test_step_four, args=(dataset_inputs,))
       return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                              axis=None)
   
    @tf.function
    def distributed_train_step_single(self, dataset_inputs):
      per_replica_losses = self.strategy.run(self.train_step_single, args=(dataset_inputs,))
      return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    @tf.function
    def distributed_test_step_single(self, dataset_inputs):
       per_replica_losses = self.strategy.run(self.test_step_single, args=(dataset_inputs,))
       return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                              axis=None)

    
    @tf.function
    def train_step(self, inputs):
        x,y,z = inputs
        with tf.GradientTape() as tape:
            y_pred = self.model([x,z], training=True)
            #y_pred = tf.squeeze(y_pred)
            loss_value = self.compute_loss(y, y_pred, self.model.losses)
            if self.args.logits:
                y_pred = tf.nn.softmax(y_pred, axis=1)
            self.train_acc_metric.update_state(y, y_pred)
            if self.args.class_weights:
                loss_value = tf.reduce_mean(self.class_weights*loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return loss_value

    @tf.function
    def test_step(self, inputs):
        x,y,z = inputs
        y_pred = self.model([x,z], training=False)
        #y_pred = tf.squeeze(y_pred)
        loss_value = self.compute_loss(y, y_pred, self.model.losses)
        if self.args.class_weights:
            loss_value = tf.reduce_mean(self.class_weights*loss_value)
        if self.args.logits:
            y_pred = tf.nn.softmax(y_pred, axis=1)
        self.val_acc_metric.update_state(y, y_pred)
        return loss_value
    
    @tf.function
    def train_step_single(self, inputs):
        x,y = inputs
        with tf.GradientTape() as tape:
            y_pred = self.model([x], training=True)
            loss_value = self.compute_loss(y, y_pred, self.model.losses)
            self.train_acc_metric.update_state(y, y_pred)
            if self.args.class_weights:
                loss_value = tf.reduce_mean(self.class_weights*loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return loss_value

    @tf.function
    def test_step_single(self, inputs):
        x,y = inputs
        y_pred = self.model([x], training=False)
        loss_value = self.compute_loss(y, y_pred, self.model.losses)
        if self.args.class_weights:
            loss_value = tf.reduce_mean(self.class_weights*loss_value)
        if self.args.logits:
            y_pred = tf.nn.softmax(y_pred, axis=1)
        self.val_acc_metric.update_state(y, y_pred)
        return loss_value
    
    @tf.function
    def train_step_four(self, inputs):
        x,y,z,u,w = inputs
        with tf.GradientTape() as tape:
            y_pred = self.model([x], training=True)
            loss_value = self.compute_loss(y, y_pred, self.model.losses)
            self.train_acc_metric.update_state(y, y_pred)
            if self.args.class_weights:
                loss_value = tf.reduce_mean(self.class_weights*loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return loss_value

    @tf.function
    def test_step_four(self, inputs):
        x,y,z,u,w  = inputs
        y_pred = self.model([x], training=False)
        loss_value = self.compute_loss(y, y_pred, self.model.losses)
        if self.args.class_weights:
            loss_value = tf.reduce_mean(self.class_weights*loss_value)
        if self.args.logits:
            y_pred = tf.nn.softmax(y_pred, axis=1)
        self.val_acc_metric.update_state(y, y_pred)
        return loss_value
    @tf.function
    def __predict(self, x):
        val_pred = self.model(x, training=False)
        return val_pred

    
    def tensorflow_analysis(self):
        for x_batch_val, y_batch_val, z_batch_val, u_batch_val, w_batch_val in self.Validation_data:
            y_pred = self.__predict([[x_batch_val, z_batch_val], [u_batch_val, w_batch_val]])

            self.val_acc_metric.update_state(y_batch_val, y_pred)
            

        m = self.val_acc_metric.result()
        
        print(m)
        self.val_acc_metric.reset_states()

        return m
    
    

    def create_training_logs(self, training, validation, train_acc, val_acc):
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "Training_logs_{}_{}_{}_{}.xlsx".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        plt1 = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "Training_{}_{}_{}_{}.png".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        plt2 = os.path.join(self.args.post_analysis_folder, str(self.args.test_fold), "Accuracy_{}_{}_{}_{}.png".format(
            self.args.connection_type, self.args.test_fold, self.args.repeat, self.args.img_size))
        data = pd.DataFrame()
        ep = [i for i in range(1, len(training)+1)]
        train_acc = train_acc[:len(training)]
        val_acc = val_acc[:len(training)]
        data["epochs"] = ep
        data["training loss"] = training
        data["training accuracy"] = train_acc
        data["valiation loss"] = validation
        data["validation accuracy"] = val_acc
        
        data.to_excel(file_p)
        fig, ax = plt.subplots()
        ax.plot(ep, training,  label="training loss")
        ax.plot(ep, validation,label="validation loss")
        ax.set(xlabel='epochs', ylabel='loss',
               title='Training loss for {}'.format(self.args.connection_type))
        ax.legend()
        fig.savefig(plt1)
        plt.close()
        
        fig, ax = plt.subplots()
        ax.plot(ep, train_acc,  label="training acc")
        ax.plot(ep, val_acc, label="validation acc")
        ax.set(xlabel='epochs', ylabel='accuracy',
               title='Training accuarcy for {}'.format(self.args.connection_type))
        ax.legend()
        fig.savefig(plt2)
        plt.close()

        return None

    def Train_model(self):
        count = 0
        Training_loss = []
        Validation_loss = []
        if self.args.features:
            self.df = self.test_data.get_dataframe()
        epo = []
        
        metrics_names = ['train_loss', "train_accuracy"]
        self.class_weights = np.zeros((len(self.args.prediction_classes)))
        for option in range(0, len(self.args.prediction_classes)):
            self.class_weights[option] = sum(list(self.df_train[self.args.prediction_classes[option]]))/self.df_train.shape[0]
        print("Classwieghts:", self.class_weights)
        for epoch in range(self.args.max_epochs):
            epo.append(epoch)
            validation_loss_value = []
            train_loss_value = []
            self.Schedule.on_epoch_begin(epoch, self.model.optimizer)
            start_time = time.time()
            pb_i = Progbar(self.num_batches, stateful_metrics=metrics_names)
            if self.args.num_input_objects == 2:
                for x,y,z in self.train_dist_dataset:
                    if self.args.model_type == 'Segmentation':
                        loss_value = self.distributed_train_step([x,y,z])
                    else:
                        loss_value = self.distributed_train_step([x,y,z])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value), ("train_accuracy", self.train_acc_metric.result())]
                    pb_i.add(self.args.batch_size, values=values)
            else:
                for x,y in self.train_dist_dataset:
                    loss_value = self.distributed_train_step_single([x,y])
                    train_loss_value.append(loss_value)
                    values = [('train_loss', loss_value)]
                    pb_i.add(self.args.batch_size, values=values)
                    
            # Display metrics at the end of each epoch.
            Training_loss.append(np.mean(train_loss_value))
            self.Train_acc[epoch] = self.train_acc_metric.result()
            #print('Epoch: %d, accuracy: %f, train_loss: %f.'%(epoch, self.train_acc_metric.result(), Training_loss[epoch]))
            # Reset training metrics at the end of each epoch
            

            # Run a validation loop at the end of each epoch.
            if self.args.num_input_objects == 2:
                for x,y,z in self.val_dist_dataset:
                    loss_value = self.distributed_test_step([x,y,z])
                    validation_loss_value.append(loss_value)
            else:
                for x,y in self.val_dist_dataset:
                    loss_value = self.distributed_test_step_single([x,y])
                    validation_loss_value.append(loss_value)
            Validation_loss.append(np.mean(validation_loss_value))
            self.Val_acc[epoch] = self.val_acc_metric.result()
            
            
            if self.args.features:
                create_features = self.Extract_features(epoch=epoch)
            if self.args.lr_schedule == "Cosine":
                self.Schedule.on_epoch_end(epoch, self.model.optimizer, Validation_loss[-1])
            else:
                self.Schedule.on_epoch_end(epoch, Validation_loss[-1])
            print("Val loss:{} Val accuracy:{} ".format(Validation_loss[-1], self.Val_acc[epoch]))
            if np.argmax(self.Val_acc) == epoch or epoch == 0:
                
                self.Save_model_weight()
                print("Weights Saved")
                count = 0

                cut_off = 2
                print("End Count: {} Cut off at: {}".format(count, cut_off))
            else:
                print("End Count: {} Cut off at: {}".format(count, cut_off))
                if epoch > self.args.cooldown:
                    count += 1

            #self.distributed_train_step.on_epoch_end(epoch)
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
            if count == cut_off:
                if self.args.early_finish == "True":
                    print("breaking loop")
                    break
        self.create_training_logs(Training_loss, Validation_loss, self.Train_acc, self.Val_acc)
    
    def predict_model(self):
        self.class_weights = np.zeros((len(self.args.prediction_classes)))
        for option in range(0, len(self.args.prediction_classes)):
            self.class_weights[option] = sum(list(self.df_train[self.args.prediction_classes[option]]))/self.df_train.shape[0]
        print("Classwieghts:", self.class_weights)
        if self.args.num_input_objects == 2:
            for x,y,z in self.val_dist_dataset:
                loss_value = self.distributed_test_step([x,y,z])

        else:
            for x,y in self.val_dist_dataset:
                loss_value = self.distributed_test_step_single([x,y])
                    
        return self.val_acc_metric.result()


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
    
    
    def run_hi_res_cam2(self, pred_index):
        
        self.model.layers[-1].activation = None
        model_layer_list = list(self.model.layers)
        for lay in range(0, len(model_layer_list)):
            if "Activation" in str(model_layer_list[lay]):
                top_conv = self.model.layers[lay]
                break

        IMAGE_SAVE = os.path.join(
            self.args.post_analysis_folder, str(self.args.val_fold))
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

            cv2.imwrite(OUTPUT_PATH, jet_heatmap)
            #cv2.imwrite(OUTPUT_PATH, heatmap)
            #superimposed_img.save(OUTPUT_PATH)
            print("you are on item {}".format(q))
            q += 1
        return IMG_NAME



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
