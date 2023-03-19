import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
from model import SegModel
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
#import keras.utils.visualize_util as vis_util
from keras import optimizers

#from utils.SegDataGenerator import *
import time

            
def train(opt, train_img, train_mask, test_img, test_mask):  
    
    #####Compile mode####
    input_size=(opt.row,opt.col,opt.ch)
    SegM=SegModel(input_size)
    model=SegM.model
    _adam=optimizers.Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='binary_crossentropy',optimizer = _adam,metrics=['accuracy'])
    ###################load data###########
    img_dim=(opt.row,opt.col)
    ####check point####
    model_checkpoint = ModelCheckpoint(opt.chekp+'.hdf5', monitor='val_acc',verbose=1, save_best_only=True)

    ###################train###########
    hist=model.fit(train_img, train_mask,validation_data=(test_img,test_mask), batch_size=opt.batch_size, epochs=opt.epochs, verbose=1,callbacks=[model_checkpoint])
 #    ######evaluate#####
    model.load_weights(opt.chekp+'.hdf5')
    y_pred=model.predict(test_img)
    f=fscore(y_pred,test_img,test_mask)
    print(f)
    return hist,f 

def fscore(tp,Images,Masks):
        

    total=0
    i=0
    fs=0
    for i in range(len(Images)):
        total += 1


        tp[i][tp[i]>0.5]=1
        tp[i][tp[i]<0.5]=0

        pred = img_to_array(tp[i]).astype(int)
        label = img_to_array(np.squeeze(Masks[i], axis=2)).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)

        fs += f1_score(flat_label,flat_pred, average='micro')

    fs=fs/total

    return fs  
            
            
if __name__=='__main__':

    train(opt)            