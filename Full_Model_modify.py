from keras.layers import BatchNormalization, Conv2D, Input, \
                         Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         AveragePooling2D,GlobalAveragePooling2D,concatenate,add, \
                         ReLU, DepthwiseConv2D
from keras.models import Model
from keras import optimizers
from Segmentation.Seg_model_modify import SegModel
import os
import sys

class FullModel():
    def __init__(self, input_size, num_class):
        self.input_size = input_size
        self.num_class = num_class
        self._build_model()
    
    def _build_model(self):
        SegM = SegModel(self.input_size)
        SegM = SegM.model
        checkpoint_filepath = '/content/drive/MyDrive/Programming/CV_project_HGR/checkpoint/checkfile_modify'
        SegM.load_weights(checkpoint_filepath)

        ## disable the train layer in SegMode

        for layer in SegM.layers[:len(SegM.layers)]:
            layer.trainable = False
        
        inp = Input(shape=self.input_size)
        inp_stream1 = SegM.input # for appearance stream
        inp_stream2 = SegM.output # for shape stream


        #### Build stream 1 ####
        # x1 = Conv2D(16, 3, activation = 'relu', padding = 'same' ,dilation_rate=1,name='CV1')(inp_stream1)
        # x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        # x1 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV2')(x1)
        # x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        # x1 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV3')(x1)
        # x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        # x1 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=3,name='CV4')(x1)

        ##############################################
        #### Build stream 1 (Depthwise Separable) ####
        ##############################################
        x1 = Conv2D(16, 3, activation = 'relu', padding = 'same' ,dilation_rate=1,name='CV1')(inp_stream1)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        ## Depthwise Separable Block 1
        x1 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS1')(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)
        x1 = Conv2D(32, 1, padding = 'same', use_bias=False)(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)

        x1 = MaxPooling2D(pool_size=(3, 3))(x1)


        ## Depthwise Separable Block 2
        x1 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS2')(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)
        x1 = Conv2D(64, 1, padding = 'same', use_bias=False)(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)

        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        ## Depthwise Separable Block 3
        x1 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS3')(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)
        x1 = Conv2D(128, 1, padding = 'same', use_bias=False)(x1)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU(max_value=6)(x1)
        
        #xf1 = MaxPooling2D(pool_size=(3, 3))(x1)
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Dense(64)(x1)
        x1 = Dropout(0.2)(x1)
        xf1 = Dense(64)(x1)



        #### Build stream 2 ####
        # x2 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=1,name='CV11')(inp_stream2)
        # x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        # x2 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV21')(x2)
        # x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        # x2 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=2 ,name='CV31')(x2)
        # x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        # x2 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=3,name='CV41')(x2)
        ##############################################
        #### Build stream 2 (Depthwise Separable) ####
        ##############################################
        x2 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=1,name='CV11')(inp_stream2)
        x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        ## Depthwise Separable Block 1
        x2 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS11')(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)
        x2 = Conv2D(32, 1, padding = 'same', use_bias=False)(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)

        x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        ## Depthwise Separable Block 2
        x2 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS21')(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)
        x2 = Conv2D(64, 1, padding = 'same', use_bias=False)(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)

        x2 = MaxPooling2D(pool_size=(3, 3))(x2)

        ## Depthwise Separable Block 3
        x2 = DepthwiseConv2D(3, strides=1, use_bias=False, padding= 'same', name='DS31')(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)
        x2 = Conv2D(128, 1, padding = 'same', use_bias=False)(x2)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU(max_value=6)(x2)

        #xf2 = MaxPooling2D(pool_size=(3, 3))(x2)
        x2 = GlobalAveragePooling2D()(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(64)(x2)
        x2 = Dropout(0.2)(x2)
        xf2 = Dense(64)(x2)

        ## Merge two stream ##
        f = add([xf1,xf2])
        prediction = Dense(self.num_class, activation="softmax")(f)

        model_final = Model(inputs=SegM.input,outputs=prediction)
        model_final.summary()
        
        self.model_Final = model_final

