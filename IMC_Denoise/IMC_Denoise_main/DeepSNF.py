# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import os
import gc

from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

from .DIMR import DIMR
from .DeepSNF_model import DeepSNF_net
from .loss_functions import create_I_divergence, create_mse
from ..DeepSNF_utils.DeepSNF_TrainGenerator import DeepSNF_Training_DataGenerator, DeepSNF_Validation_DataGenerator
from ..Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_exact_unbiased, Anscombe_inverse_direct

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class DeepSNF():
    
    # DeepSNF class, including DeepSNF training, prediction, etc.
    
    def __init__(self, train_epoches = 200, train_learning_rate = 0.001, lr_decay_rate = 0.6, train_batch_size = 128, mask_perc_pix = 0.2, val_perc = 0.1,
                 loss_func = "I_divergence", weights_name = None, loss_name = None, weights_dir = None, is_load_weights = False, 
                 truncated_max_rate = 0.99999, lambda_HF = 0):
        
        """
        Parameters
        ----------
        train_epoches : int, optional
            The default is 100.
        train_learning_rate : float, optional
            The default is 0.001.
        lr_decay_rate: float, optional
            The default is 0.6.
        train_batch_size : int, optional
            The default is 256.
        mask_perc_pix : float, optional
            Percentage of the masked pixels for every patch. The default is 0.2.
        val_perc : float, optional
            Percentage of the training set as the validation set. The default is 0.15.
        loss_func : "mse", "mse_relu" or "I_divergence", optional
            If "I_divergence", the framework will be DeepSNF;
            If "mse", the framework will be Noise2Void;
            If "mse_relu", the framework will be Noise2Void with Anscombe transformation and Relu activation
            The default is "I_divergence".
        weights_name : string, optional
            The file name of the saved weights. .hdf5 format. The default is None.
        loss_name : string, optional
            The file name of the saved losses. .npz or .mat format. The default is None.
        weights_dir : string, optional
            The directory used for saving weights file. The default is None.
        is_load_weights : bool, optional
            True: load pre-trained weights file from disk for transfer learning and prediction.
            False: not load any pre-trained weights. 
            The default is False.
        truncated_max_rate: float, optional
            The max_val of the channel is 1.1*max(images, truncated_max_rate*maximum).
            The default is 0.99999. It should work in most cases.
            When the maximum of the predicted image is much higher, the value may be set higher during 
            training. But the values which is out of the range of the training set may not be predicted
            well. Therefore, the selection of a good training set is important.
        lambda_HF: float, optional
            The parameter for Hessian regularization. The best value is around 3e-6.

        """
        if not isinstance(train_epoches, int):
            raise ValueError('The train_epoches must be an integer!')
        self.train_epoches = train_epoches
        
        self.train_learning_rate = train_learning_rate
        
        self.lr_decay_rate = lr_decay_rate
        
        if not isinstance(train_batch_size, int):
            raise ValueError('The train_batch_size must be an integer!')
        self.train_batch_size = train_batch_size
        
        assert mask_perc_pix >=0 and mask_perc_pix <= 100, "mask_perc_pix must be between 0 and 100!"
        self.mask_perc_pix = mask_perc_pix
        
        assert val_perc >=0 and val_perc <= 1, "val_perc must be between 0 and 1!"
        self.val_perc = val_perc
        
        assert truncated_max_rate >=0 and truncated_max_rate <= 1, "truncated_max_rate must be between 0 and 1!"
        self.truncated_max_rate = truncated_max_rate
        
        self.loss_function = loss_func
        self.loss_name = loss_name
        
        if self.loss_function == "mse_relu":
            self.min_val = 2*np.sqrt(3/8)
        else:
            self.min_val = 0
        
        assert isinstance(is_load_weights, bool), "is_load_weights must be a bool value!"
        self.is_load_weights = is_load_weights
        
        if weights_dir is not None:
            self.weights_dir = weights_dir
        else:
            self.weights_dir = os.path.abspath('trained_weights')
            if not os.path.exists(self.weights_dir):
                os.makedirs(self.weights_dir)
        
        self.weights_name = weights_name
        if self.weights_name is not None:
            if not self.weights_name.endswith('.hdf5'):
                print('the weights file should end with .hdf5!')
                return
            
        self.lambda_HF = lambda_HF
        
        if is_load_weights:
            self.trained_model = self.load_model()
    
    def buildModel(self, input_dim):
        
        """
        Define the Model building for an arbitrary input size

        """
        input_ = Input (shape = (input_dim))
        if self.loss_function == "mse_relu": 
            network_name = 'N2V_Anscombe_relu_'
        elif self.loss_function == "mse": 
            network_name = 'N2V_'
        else:
            self.loss_function == "I_divergence"
            network_name = 'DeepSNF_'
            
        act_ = DeepSNF_net(input_, network_name, loss_func = self.loss_function, trainable_label = True)
        model = Model (inputs= input_, outputs=act_)  
        
        opt = optimizers.Adam(lr=self.train_learning_rate)
        if self.loss_function != "I_divergence":    
            model.compile(optimizer=opt, loss = create_mse(lambda_HF = self.lambda_HF))
        else:
            model.compile(optimizer=opt, loss = create_I_divergence(lambda_HF = self.lambda_HF))
            
        return model
    
    def train(self, X):
        
        """
        Train a DeepSNF model for a specific marker.

        Parameters
        ----------
        X : float
            training set.

        Returns
        -------
        Training and validation losses.

        """
        if X.ndim != 3:
            print('Please check the input data, must be 3d [N*R*C]!')
            return
        
        if self.is_load_weights:
            model = self.trained_model
            if self.loss_function == "mse_relu":
                X = Anscombe_forward(X)
            X = self.__Normalize__(X, self.range_val, self.min_val)
        else:
            model = self.buildModel((None, None, 1)) 
            if self.loss_function == "mse_relu":
                X = Anscombe_forward(X)
            X, self.range_val = self.normalize_patches(X)  
        
        print('The range value to the corresponding model is {}.'.format(self.range_val))
        
        if not self.is_load_weights:
            X[X > 1.0] = 1.0
            X[X < 0.0] = 0.0
        X = np.expand_dims(X, axis = -1)
        
        print('Input Channel Shape => {}'.format(X.shape))
          
        X_train, X_test = train_test_split(X, test_size = self.val_perc, random_state = 42)
        del X
        print('Number of Training Examples: %d' % X_train.shape[0])
        print('Number of Validation Examples: %d' % X_test.shape[0])
        
        STEPS_PER_EPOCH = int(np.floor(X_train.shape[0]/self.train_batch_size)+1)
        # Setting type
        X_train = X_train.astype('float32') 
        X_test = X_test.astype('float32')
        p_row_size, p_col_size =  X_train.shape[1:-1]
        
        # loss history recorder
        history = LossHistory()
        
        # Change learning when loss reaches a plataeu
        change_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = self.lr_decay_rate, patience = 20, min_lr = 0.0000005, verbose = 1)
        
        # Save the model weights after each epoch
        if self.weights_name is not None: 
            np.savez(os.path.join(self.weights_dir, self.weights_name.replace('.hdf5','_range_val.npz')), range_val = self.range_val)
            checkpointer = ModelCheckpoint(filepath = os.path.join(self.weights_dir, self.weights_name), verbose = 1, save_best_only = False)
            callback_list = [history, checkpointer, change_lr]
        else:
            callback_list = [history, change_lr]
           
        training_data = DeepSNF_Training_DataGenerator(X_train, self.train_batch_size, self.mask_perc_pix, (p_row_size, p_col_size))
        X_test, Y_test = DeepSNF_Validation_DataGenerator(X_test, self.mask_perc_pix, (p_row_size, p_col_size))
    
        # Inform user training begun
        print('Training model...')
        
        train_history = model.fit_generator(generator = training_data, \
                                            steps_per_epoch = STEPS_PER_EPOCH, epochs = self.train_epoches, verbose=1, \
                                            validation_data = (X_test, Y_test), \
                                            callbacks = callback_list)    
    
        # Inform user training ended
        print('Training Completed!')
        
        self.trained_model = model
    
        # plot the loss function progression during training
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        
        # Save datasets to a matfile to open later in matlab
        if  self.loss_name is not None:
            if self.loss_name.endswith('.npz'):
                np.savez(os.join.path(self.weights_dir, self.loss_name), train_loss = loss, val_loss = val_loss)
            elif self.loss_name.endswith('.mat'):
                sio.savemat(os.join.path(self.weights_dir, self.loss_name), {"train_loss": loss, "val_loss": val_loss})
            else:
                print('saved format should be .npz or .mat. Save failed.')
        
        return loss, val_loss
    
    def load_model(self):
        
        """
        Load pre-trained model from disk.

        """
        if self.weights_name is None:
            print('When loading a model, a legal .hdf5 file must be provided. Weights loaded failed!')
            return
        
        # Build the DeepSNF network structure
        model = self.buildModel((None, None, 1))
        
        # Load the trained weights
        load_weights_name = os.join.path(self.weights_dir, self.weights_name)
        if os.path.exists(load_weights_name): 
            model.load_weights(load_weights_name)
            print('Pre-trained model {} loaded successfully.'.format(load_weights_name))
        else:
            print('Pre-trained model loaded fail. Please check if the file {} exists.'.format(load_weights_name))
            return
           
        load_range_name = os.path.join(self.weights_dir, self.weights_name.replace('.hdf5', '_range_val.npz'))
        if os.path.exists(load_range_name): 
            loaded_range_val = np.load(load_range_name)
            self.range_val = loaded_range_val['range_val']
            print('Pre-calculated range value file {} loaded successfully.'.format(load_range_name))
        else:
            print('Pre-calculated range value file loaded fail. Please check if the file {} exists.'.format(load_range_name))
            return
        
        return model
    
    def normalize_patches(self, patches):
        
        """
        Normalize the training set.

        """
        patches_sort = np.sort(patches, axis = None)
        truncated_maxval = patches_sort[int(self.truncated_max_rate*np.shape(patches_sort)[0])] #0.99999
        max_val = 1.1*truncated_maxval
        range_val = max_val - self.min_val
        patches_normalized = self.__Normalize__(patches, range_val, self.min_val)
        
        return patches_normalized, range_val
    
    def predict(self, X):
        
        """
        Using trained model to denoise a single IMC image for a specific marker channel
    
        Parameters
        ----------
        X : list(array)
            Hot pixel-removed IMC images.
    
        Returns
        -------
        Predicted images.
    
        """
        
        if X.ndim != 2:
            raise Exception("For DeepSNF deneising, the input must be a 2d image!")
            
        if self.range_val is None:
            print('In prediction, range value for the marker channel must be defined!')
            return
        
        # Normalize the input image
        Input_img_norm = self.__Normalize__(X, self.range_val, self.min_val)
        Input_img_norm[Input_img_norm < 0.0] = 0.0 
         
        # Pad image to suppress batch effect         
        Rows, Cols = np.shape(Input_img_norm)
        Rows_new = int((Rows//16+1) * 16)
        Cols_new = int((Cols//16+1) * 16)   
        Rows_diff = Rows_new - Rows
        Cols_diff = Cols_new - Cols
            
        Rows_diff1, Rows_diff2 = self.__split_border__(Rows_diff)
        Cols_diff1, Cols_diff2 = self.__split_border__(Cols_diff)
                 
        Input_img_pad = np.pad(Input_img_norm,((Rows_diff1,Rows_diff2),(Cols_diff1,Cols_diff2)),'edge')   
        Input_img_pad_dims = np.expand_dims(Input_img_pad,axis=-1) 
        Input_img_pad_dims = np.expand_dims(Input_img_pad_dims,axis=-1)
        Input_img_pad_dims = np.transpose(Input_img_pad_dims, (2, 0, 1, 3))
                               
        # Setting type
        Input_channel = Input_img_pad_dims.astype('float32')
            
        # Make a prediction
        predicted = self.trained_model.predict_on_batch(Input_channel)
        _ = gc.collect()
        # K.clear_session()
        predicted = predicted[0,Rows_diff1:(-Rows_diff2),Cols_diff1:(-Cols_diff2),0]
        predicted_img = self.__Denormalize__(predicted, self.range_val, self.min_val)
        
        return predicted_img
    
    def predict_batch(self, X):
        
        """
        Using trained model to denoise a batch of IMC images with the same shape for a specific marker channel
    
        Parameters
        ----------
        X : list(array)
            Hot pixel-removed IMC images.
    
        Returns
        -------
        Predicted images.
    
        """
        if X.ndim != 3:
            raise Exception("For DeepSNF batch processing, the input must be a 3d array ([N, R, C])!")
            
        if self.range_val is None:
            print('In prediction, range values for the marker channel must be defined!')
            return
            
        # Normalize the input image
        Input_img_norm = self.__Normalize__(X, self.range_val, self.min_val)
        Input_img_norm[Input_img_norm < 0.0] = 0.0 
         
        # Pad image to suppress batch effect         
        Nums, Rows, Cols = np.shape(Input_img_norm)
        Rows_new = int((Rows//16+1) * 16)
        Cols_new = int((Cols//16+1) * 16)   
        Rows_diff = Rows_new - Rows
        Cols_diff = Cols_new - Cols
            
        Rows_diff1, Rows_diff2 = self.__split_border__(Rows_diff)
        Cols_diff1, Cols_diff2 = self.__split_border__(Cols_diff)
                 
        Input_img_pad = np.pad(Input_img_norm,((0,0), (Rows_diff1,Rows_diff2), (Cols_diff1,Cols_diff2)), 'edge')   
        Input_img_pad_dims = np.expand_dims(Input_img_pad, axis=-1) 
                                      
        # Setting type
        Input_channel = Input_img_pad_dims.astype('float32')
            
        # Make a prediction
        predicted = self.trained_model.predict(Input_channel)
        _ = gc.collect()
        # K.clear_session()
        predicted = predicted[:,Rows_diff1:(-Rows_diff2),Cols_diff1:(-Cols_diff2),0]
        predicted_img = self.__Denormalize__(predicted, self.range_val, self.min_val)
         
        return predicted_img
    
    def perform_DeepSNF(self, X):
        
        """
        Perform DeepSNF for a single image.

        Parameters
        ----------
        X : array(int or float)
       
        Returns
        -------
        X_denoised : float
           
        """
        if self.loss_function == "mse_relu":
            X = Anscombe_forward(X)
            
        X_denoised = self.predict(X)
        
        if self.loss_function == "mse_relu":
            X_denoised = Anscombe_inverse_exact_unbiased(X_denoised)
            
        return X_denoised
    
    def perform_DeepSNF_batch(self, X):
        
        """
        Perform DeepSNF for a batch of images with the same shape.

        Parameters
        ----------
        X : array(int or float)
       
        Returns
        -------
        X_denoised : float
           
        """
        if self.loss_function == "mse_relu":
            X = Anscombe_forward(X)
            
        X_denoised = self.predict_batch(X)
        
        if self.loss_function == "mse_relu":
            X_denoised = Anscombe_inverse_exact_unbiased(X_denoised)
            
        return X_denoised
    
    def perform_IMC_Denoise(self, X, n_neighbours = 4, n_iter = 3, window_size = 3):
        
        """
        Perform IMC_Denoise for a single image.

        Parameters
        ----------
        X : array(int or float)
       
        Returns
        -------
        X_denoised : float

        """
        X_Anscombe_transformed = Anscombe_forward(X)
        X_DIMR = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size).predict_augment(X_Anscombe_transformed)
        if self.loss_function == "mse_relu":
            X_SNF = self.predict(X_DIMR)
            X_denoised = Anscombe_inverse_exact_unbiased(X_SNF)
        else:
            X_DIMR = Anscombe_inverse_direct(X_DIMR)
            X_denoised = self.predict(X_DIMR)
        
        return X_denoised
    
    def perform_IMC_Denoise_batch(self, X, n_neighbours = 4, n_iter = 3, window_size = 3):
        
        """
        Perform IMC_Denoise for a batch of images with the same shape.

        Parameters
        ----------
        X : array(int or float)
       
        Returns
        -------
        X_denoised : float

        """
        if X.ndim != 3:
            raise Exception("For DIMR batch processing, the input must be a 3d array ([N, R, C])!")
        
        X_Anscombe_transformed = Anscombe_forward(X)
        dimr = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = 3)
        X_DIMR = np.zeros(np.shape(X_Anscombe_transformed))
        for ii in range(np.shape(X_Anscombe_transformed)[0]):
            X_DIMR[ii, :, :] = dimr.predict_augment(X_Anscombe_transformed[ii, :, :])
        
        if self.loss_function == "mse_relu":
            X_SNF = self.predict_batch(X_DIMR)
            X_denoised = Anscombe_inverse_exact_unbiased(X_SNF)
        else:
            X_DIMR = Anscombe_inverse_direct(X_DIMR)
            X_denoised = self.predict(X_DIMR)
        
        return X_denoised
    
    def __Normalize__(self, img, range_val, min_val):
        return np.divide(img - min_val, range_val)
    
    def __Denormalize__(self, img, range_val, min_val):
        return img * range_val + min_val
    
    def __split_border__(self, length):
        half_length = int(length/2)
        if length%2 == 0:
            return half_length, half_length
        else:
            return half_length, half_length + 1
