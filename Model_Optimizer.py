import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.callbacks import Callback

import math
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, args):
        self.args = args
        self.T_max = self.args.max_epochs/3
        self.eta_max = self.args.Max_lr
        self.eta_min = self.args.Min_lr
        self.verbose = 1
        self.warmup_epochs = self.args.warmup_epochs

    def on_epoch_begin(self, epoch, mod_optimizer , logs=None):
        if not hasattr(mod_optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.warmup_epochs > epoch and epoch < self.args.max_epochs - self.args.lr_cooldown:
            diff = (epoch/self.warmup_epochs)*self.eta_max
            lr = diff + self.eta_min
        elif self.warmup_epochs <= epoch and epoch < self.args.max_epochs - self.args.lr_cooldown:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        elif self.warmup_epochs <= epoch and epoch >= self.args.max_epochs - self.args.lr_cooldown:
            lr = self.eta_min*((self.args.max_epochs-epoch)+1)
        K.set_value(mod_optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, mod_optimizer, loss, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(mod_optimizer.lr)
        


        



@keras_export('keras.callbacks.ReduceLROnPlateau')
class CustomSchedule(Callback):
  def __init__(self,
              ## Custom modification:  Deprecated due to focusing on validation loss
              # monitor='val_loss',
              args= None,
              factor=0.9,
              patience=5,
              verbose=0,
              mode='auto',
              min_delta=1e-4,
              cooldown=0,
              min_lr=0.00000001,
              sign_number = 4,
              ## Custom modification: Passing optimizer as arguement
              optim_lr = None,
              ## Custom modification:  linearly reduction learning
              reduce_lin = False,
              **kwargs):

    ## Custom modification:  Deprecated
    # super(ReduceLROnPlateau, self).__init__()

    ## Custom modification:  Deprecated
    # self.monitor = monitor
    print(optim_lr)
    ## Custom modification: Optimizer Error Handling
    if tf.is_tensor(optim_lr) == False:
        raise ValueError('Need optimizer !')
    if factor >= 1.0:
        raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    ## Custom modification: Passing optimizer as arguement
    self.optim_lr = optim_lr  
    self.args = args
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self.sign_number = sign_number
    

    ## Custom modification: linearly reducing learning
    self.reduce_lin = reduce_lin
    self.reduce_lr = False
    

    self._reset()

  def _reset(self):
      if self.mode not in ['auto', 'min', 'max']:
          print('Learning Rate Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
          self.mode = 'auto'
      if (self.mode == 'min' or
        ## Custom modification: Deprecated due to focusing on validation loss
        # (self.mode == 'auto' and 'acc' not in self.monitor)):
            (self.mode == 'auto')):
          self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
          self.best = np.Inf
      else:
          self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
          self.best = -np.Inf
      self.cooldown_counter = 0
      self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, loss, logs=None):


    logs = logs or {}
    ## Custom modification: Optimizer
    # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
    # and therefore can be modified to          
    logs['lr'] = float(self.optim_lr.numpy())

    ## Custom modification: Deprecated due to focusing on validation loss
    # current = logs.get(self.monitor)

    current = float(loss)

    ## Custom modification: Deprecated due to focusing on validation loss
    # if current is None:
    #     print('Reduce LR on plateau conditioned on metric `%s` '
    #                     'which is not available. Available metrics are: %s',
    #                     self.monitor, ','.join(list(logs.keys())))

    # else:

    if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

    if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
    elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
            ## Custom modification: Optimizer Learning Rate
            # old_lr = float(K.get_value(self.model.optimizer.lr))
            old_lr = float(self.optim_lr.numpy())
            if old_lr > self.min_lr and self.reduce_lr == True:
                ## Custom modification: Linear learning Rate
                    new_lr = old_lr * self.factor
                    ## Custom modification: Error Handling when learning rate is below zero
                    if new_lr <= 0:
                        print('Learning Rate is below zero: {}, '
                        'fallback to minimal learning rate: {}. '
                        'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))  
                    self.reduce_lr = False                           
            else:
                old_lr = float(self.optim_lr.numpy())
                new_lr = old_lr * self.factor
                #new_lr = self.optim_lr * self.factor ^ (epoch / self.args.max_epochs)
                #new_lr = self.args.Max_lr * np.exp(-1 * self.factor * (epoch / self.args.max_epochs))

                new_lr = max(new_lr, self.args.Min_lr)

                print(new_lr)
                ## Custom modification: Optimizer Learning Rate
                # K.set_value(self.model.optimizer.lr, new_lr)
                self.optim_lr.assign(new_lr)

                if self.verbose > 0:
                    print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                            'rate to %s.' % (epoch + 1, float(new_lr)))
                self.cooldown_counter = self.cooldown
                self.wait = 0
                

  def in_cooldown(self):
    return self.cooldown_counter > 0
