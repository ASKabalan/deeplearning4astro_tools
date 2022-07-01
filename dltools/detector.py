import os
import datetime
import types
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import Bunch

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.math import exp

from dltools.batch import BatchGeneratorBuilder
from dltools.metric import iou


class ObjectDetector(object):
    """Object detector.

    Parameters
    ----------
    batch_size : int, optional
        The batch size used during training. Set by default to 32 samples.

    epoch : int, optional
        The number of epoch for which the model will be trained. Set by default
        to 50 epochs.

    model_check_point : bool, optional
        Whether to create a callback for intermediate models.
    
    nb_examples : int, optional (default 1)
        size of the augmented batch size (1 is same as train 2 is double)

    Attributes
    ----------
    model_ : object
        The DNN model.

    params_model_ : Bunch dictionary
        All hyper-parameters to build the DNN model.

    """

    def __init__(self, model, lr=1e-4, batch_size=32, epoch=2,  model_check_point=True):
        self.model_ = self._build_model(model, lr)
        self.params_model_ = self._init_params_model()
        self.batch_size = batch_size
        self.learning_rate = lr
        self.epoch = epoch
        self.log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_check_point = model_check_point


    def set_log_name(self,log_name):
        self.log_name = log_name
    """
        call use_learning_scheduler to activate LearningRateScheduler
        @param : lamda:
                 is the function used to change lr in function of the epoch
    """
    def use_learning_scheduler(self,lamda):

        assert(isinstance(lamda,types.FunctionType))

        self.params_model_.use_learning_scheduler = True
        self.params_model_.lr_scheduler_lamda     = lamda

    """
        call use_early_stop to activate EarlyStopping
        @param : patience:
                 The number of epochs before stopping
               : min_delta:
                 The tolerance of non-improvement
    """
    def use_early_stop(self,patience=10,min_delta=0.001):
        self.params_model_.early_stopping         = True
        self.params_model_.es_patience            = patience
        self.params_model_.es_min_delta           = min_delta

    """
        call use_reduce_lr to activate ReduceLROnPlateau
        @param : patience:
                 The number of epochs before stopping
               : min_delta:
                 The tolerance of non-improvement
               : factor:
                 The factor of the reduction IE : lr = factor * lr
    """
    def use_reduce_lr(self,patience=5,factor=0.2,min_delta=0.001):
        self.params_model_.reduce_learning_rate   = True
        self.params_model_.lr_patience            = patience
        self.params_model_.lr_factor              = factor
        self.params_model_.lr_min_delta           = min_delta
    
    """
        set_augmentation_parameters control the augmentation parameters
        @param : flip:
                 flip images (either according to X axis or Y axis)
               : rot:
                 Rotate the images (either 90 or -90 degrees)
               : roll:
                 Shift the images by 30 (either to the left or right) (TODO this can be catastrophic check batch.py)
    """
    def set_augmentation_parameters(self,flip=True,rot=True,roll=True,augmentation_size=1):
        self.params_model_.augment_tuple    = (flip,rot,roll)
        self.params_model_.augmented_size   = augmentation_size

    def fit(self, X, y):

        # build the box encoder to later encode y to make usable in the model
        train_dataset = BatchGeneratorBuilder(X, y
        , self.params_model_.augmented_size 
        , self.params_model_.augment_tuple
        )

        train_generator, val_generator, n_train_samples, n_val_samples = \
            train_dataset.get_train_valid_generators(
                batch_size=self.batch_size)

        # create the callbacks to get during fitting
        callbacks = self._build_callbacks()

        # fit the model
        history = self.model_.fit(
            x=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

        return history

    def predict(self, X):
        Y_p = self.model_.predict(np.expand_dims(X, -1))
        return Y_p

    def predict_score(self,X,Y):
        Y_p = self.model_.predict(np.expand_dims(X, -1))
        s = iou(Y_p.squeeze(),Y.squeeze())
        return s
    
    def plot_random_results(self, X_test, y_test, filename):
        n_gal = 5
        idx = np.random.randint(0, len(y_test), size=n_gal)
        X = X_test[idx]
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        y_true = y_test[idx]
        y_pred = self.model_.predict(X)

        titles = [
            'blend',
            'true segmentation',
            'output',
            'output thresholded',
        ]
        fig_size = (10, 12)
        fig, ax = plt.subplots(nrows=n_gal, ncols=4, figsize=fig_size)
        for i in range(n_gal):
            img = np.squeeze(X[i])
            yt = np.squeeze(y_true[i])
            yp = np.squeeze(y_pred[i])
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(yt)
            ax[i, 2].imshow(yp)
            ax[i, 3].imshow(yp.round())
            if i == 0:
                for idx, a in enumerate(ax[i]):
                    a.set_title(titles[idx])
            for a in ax[i]:
                a.set_axis_off()
        plt.savefig('{filename}'.format(filename=filename))

    @staticmethod
    def plot_history(history, filename):
        plt.semilogy(history.epoch, history.history['loss'], label='loss')
        plt.semilogy(history.epoch, history.history['val_loss'], label='val_loss')
        plt.title('Training performance')
        plt.legend()
        plt.savefig("{filename}".format(filename=filename))

    
    ###########################################################################
    # Setup model

    @staticmethod
    def _init_params_model():
        params_model = Bunch()

        # image and class parameters
        params_model.img_rows               = 128
        params_model.img_cols               = 128
        params_model.img_channels           = 1

        # architecture params   
        params_model.output_channels        = 1             # size of the output in depth
        params_model.depth                  = 16            # depth of all hidden layers
        params_model.n_layers               = 6             # number of layers before last
        params_model.conv_size0             = (3, 3)        # kernel size of first layer
        params_model.conv_size              = (3, 3)        # kernel size of intermediate layers
        params_model.last_conv_size         = (3, 3)        # kernel size of last layer
        params_model.activation             = 'relu'        # activation between layers
        params_model.last_activation        = 'sigmoid'     # final activation (sigmoid nice if binary objective)
        params_model.initialization         = 'he_normal'   # weight initialization
        params_model.constraint             = None          # kernel constraints (None, nonneg, unitnorm, maxnorm)
        params_model.dropout_rate           = 0.0           # percentage of weights not updated (0 = no dropout)
        params_model.sigma_noise            = 0.01          # random noise added before last layer (0 = no noise added)

        # optimizer parameters  
        params_model.lr                     = 1.e-4
        params_model.beta_1                 = 0.9
        params_model.beta_2                 = 0.999
        params_model.epsilon                = 1e-08
        params_model.decay                  = 5e-05

        # loss parameters   
        params_model.keras_loss             = 'binary_crossentropy'

        #augmentation parameters    
        params_model.augment_flip           = False
        params_model.augment_rot            = False
        params_model.augment_roll           = False
        params_model.augmented_size         = 1

        # callbacks parameters  
        params_model.early_stopping         = False
        params_model.es_patience            = 10
        params_model.es_min_delta           = 0.001

        # Learning rate scheduler callback
        params_model.use_learning_scheduler = False
        params_model.lr_scheduler_lamda     = lambda epoch: self.learning_rate * (1 / (epoch + 1))
        params_model.reduce_learning_rate   = False
        params_model.lr_patience            = 5
        params_model.lr_factor              = 0.2
        params_model.lr_min_delta           = 0.001
        params_model.lr_cooldown            = 2

        params_model.tensorboard = True

        return params_model

    def _build_model(self, model, lr):

        # load the parameter for the SSD model
        params_model = self._init_params_model()

        optimizer = Adam(lr=lr)

        model.compile(optimizer=optimizer, loss=params_model.keras_loss)

        return model


    def _build_callbacks(self):
        callbacks = []

        if self.model_check_point:
            callbacks.append(
                ModelCheckpoint('./fcnn_weights_best.h5',
                                monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=True,
                                period=1,
                                verbose=1))
        # add early stopping
        if self.params_model_.early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss',
                              min_delta=self.params_model_.es_min_delta,
                              patience=self.params_model_.es_patience,
                              verbose=1))

        # reduce learning-rate when reaching plateau
        if self.params_model_.reduce_learning_rate:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=self.params_model_.lr_factor,
                                  patience=self.params_model_.lr_patience,
                                  cooldown=self.params_model_.lr_cooldown,
                                  # min_delta=self.params_model_.lr_min_delta,
                                  verbose=1)
            )

        if self.params_model_.use_learning_scheduler:
            callbacks.append(
                LearningRateScheduler(self.params_model_.lr_scheduler_lamda)
            )

        if self.params_model_.tensorboard:
            log_dir = "./logs/fit/" + self.log_name
            callbacks.append(
                TensorBoard(log_dir=log_dir)
            ) 

        return callbacks