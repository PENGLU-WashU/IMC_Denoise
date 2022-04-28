# -*- coding: utf-8 -*-
from keras.layers import Activation, UpSampling2D, Convolution2D, MaxPooling2D, BatchNormalization, Dropout, concatenate, add

def conv_bn_relu_gen(nb_filter, rk, ck, st, names, trainable_label = True):
    def f(input):
        conv = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(st,st),padding="same", 
                             use_bias=False,kernel_initializer="truncated_normal",name='conv-'+names)(input)
        conv_norm = BatchNormalization(name='BN-'+names, trainable=trainable_label)(conv)
        conv_norm_relu = Activation(activation = 'relu',name='Relu-'+names)(conv_norm)
        return conv_norm_relu
    return f

def conv_bn(nb_filter, rk, ck, st, names):
    def f(input):
        conv = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(st,st),padding="same", 
                             use_bias=False,kernel_initializer="truncated_normal",name='conv-'+names)(input)
        return conv
    return f

def ResBlock(nb_filter, rk, ck, st, res_names, trainable_label):
    def f(input):
        Node1 = conv_bn_relu_gen(nb_filter, rk, ck, 1, 'Nodes1'+res_names,trainable_label)(input)
        Node2 = conv_bn(nb_filter, rk, ck, 1, 'Nodes2'+res_names)(Node1)
        Node3 = conv_bn(nb_filter, 1, 1, 1, 'Nodes3'+res_names)(input)
        Node4 = add([Node2,Node3])   
        conv_norm = BatchNormalization(name='BN_act-'+res_names, trainable=trainable_label)(Node4)
        Res = Activation(activation = 'relu',name='Nodes_act-'+res_names)(conv_norm)
        return Res
    return f

def ResBlock2(nb_filter, rk, ck, st, res_names,trainable_label):
    def f(input):
        Node1 = conv_bn_relu_gen(nb_filter, rk, ck, 1, 'Nodes1'+res_names,trainable_label)(input)
        Node2 = conv_bn(nb_filter, rk, ck, 1, 'Nodes2'+res_names)(Node1)
        Node4 = add([Node2,input])    
        conv_norm = BatchNormalization(name='BN_act-'+res_names, trainable=trainable_label)(Node4)
        Res = Activation(activation = 'relu',name='Nodes_act-'+res_names)(conv_norm)
        return Res
    return f

def DeepSNF_net(input, names, loss_func,trainable_label = True):
    filter_num = 64
    Features1 = ResBlock(filter_num, 3, 3, 1, res_names=names+'ResBlock1',trainable_label=trainable_label)(input)
    
    pool1 = MaxPooling2D(pool_size=(2,2),name=names+'Pool1')(Features1)
    Features2 = ResBlock(filter_num*2, 3, 3, 1, res_names=names+'ResBlock2',trainable_label=trainable_label)(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool2')(Features2)
    Features3 = ResBlock(filter_num*4, 3, 3, 1, res_names=names+'ResBlock3',trainable_label=trainable_label)(pool2)
    
    pool3 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool3')(Features3)
    Features4 = ResBlock(filter_num*8, 3, 3, 1, res_names=names+'ResBlock4',trainable_label=trainable_label)(pool3)
    drop2 = Dropout(0.5)(Features4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool4')(drop2)
    Features5 = ResBlock(filter_num*16, 3, 3, 1, res_names=names+'ResBlock5',trainable_label=trainable_label)(pool4)
    drop1 = Dropout(0.5)(Features5)
    
    up1 = UpSampling2D(size=(2, 2),name=names+'Upsample1')(drop1)
    merge1 = concatenate([drop2,up1], axis = 3)
    Features6 = ResBlock(filter_num*8, 3, 3, 1, res_names=names+'ResBlock6',trainable_label=trainable_label)(merge1)
    
    up2 = UpSampling2D(size=(2, 2),name=names+'Upsample2')(Features6)
    merge2 = concatenate([Features3,up2], axis = 3)  
    Features7 = ResBlock(filter_num*4, 3, 3, 1, res_names=names+'ResBlock7',trainable_label=trainable_label)(merge2)
    
    up3 = UpSampling2D(size=(2, 2),name=names+'Upsample3')(Features7)
    merge3 = concatenate([Features2,up3], axis = 3)
    Features8 = ResBlock(filter_num*2, 3, 3, 1, res_names=names+'ResBlock8',trainable_label=trainable_label)(merge3)
    
    up4 = UpSampling2D(size=(2, 2),name=names+'Upsample4')(Features8)
    merge4 = concatenate([Features1,up4], axis = 3)
    Features9 = ResBlock(filter_num, 3, 3, 1, res_names=names+'ResBlock9',trainable_label=trainable_label)(merge4)
    
    if loss_func == "I_divergence":
        Features10 = Convolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", 
                               activation="softplus", use_bias = False, 
                               kernel_initializer="truncated_normal",
                               name='Prediction_softplus')(Features9)
    elif loss_func == "mse":
        Features10 = Convolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", 
                               activation="linear", use_bias = False, 
                               kernel_initializer="truncated_normal",
                               name='Prediction_linear')(Features9)
        
    elif loss_func == "mse_relu":
        Features10 = Convolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", 
                               activation="relu", use_bias = False, 
                               kernel_initializer="truncated_normal",
                               name='Prediction_relu')(Features9)

    return Features10
