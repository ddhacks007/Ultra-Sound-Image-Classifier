import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Lambda
from tensorflow.keras.layers import Input,LeakyReLU, MaxPooling2D, Conv2D, BatchNormalization, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, DepthwiseConv2D, Multiply, Reshape, Maximum, Minimum, Subtract, SpatialDropout2D, Average

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

from tensorflow.keras import initializers
from other import *

from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import uuid
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import properties
import argparse


img_height=properties.img_h_en
img_width=properties.img_w_en
num_classes = properties.num_classes
num_workers= properties.num_workers
batch_size= properties.en_batch_size
epochs = properties.epochs
lr_init=1e-2

def conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None,act='l'):
    kr = regularizers.l2(weight_decay)
    ki = initializers.he_normal()

    x = Conv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki, strides=(stride, stride),
               use_bias=False, padding='same', kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    #
    if(act == 'm'):
        x = Mish()(x)

    if (act == 'l'):
        x = LeakyReLU(alpha=0.1)(x)

    if (act == 'r'):
        x = Activation('relu')(x)

    if (act == 's'):
        x = Activation('sigmoid')(x)

    if dropout_rate != None and dropout_rate != 0.:
        x = Dropout(dropout_rate)(x)
    return x


def separable_conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None, act='l'):

  ki = initializers.he_normal()

  kr = regularizers.l2(weight_decay)


  x = SeparableConv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki,
                      strides=(stride, stride), use_bias=False, padding='same',
                      kernel_regularizer=kr)(x)

  x = BatchNormalization()(x)

  if (act == 'm'):
    x = mish(x)

  if (act == 'l'):
    x = LeakyReLU(alpha=0.2)(x)

  if (act == 'r'):
    x = Activation('relu')(x)

  if (act == 's'):
    x = Activation('sigmoid')(x)

  if dropout_rate != None and dropout_rate != 0.:
    x = Dropout(dropout_rate)(x)
  return x


def fusion_block(tensors,name,type='add'):
    if(type=='add'):
        return Add(name='add_'+name)(tensors)

    if (type == 'max'):
        return Maximum(name='max_'+name)(tensors)

    if (type == 'con'):
        return Concatenate(name='conc_'+name)(tensors)

    if (type == 'avg'):
        return Average(name='avg_'+name)(tensors)

## Mish Activation Function
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.math.tanh(tf.math.log(1+tf.exp(x))))(x)

def activation(x, ind,t='-',n=255):

    if (t == 'r'):
        return Activation('relu',name='relu_'+str(ind))(x)
    if (t == 'l'):
        return LeakyReLU(name='leakyrelu_'+str(ind),alpha=0.2)(x)
    if (t == 'e'):
        return ELU(name='elu_'+str(ind))(x)
    if (t == 'n'):
        def reluclip(x, max_value=n):
            return K.relu(x, max_value=max_value)
        return Lambda(function=reluclip)(x)
    if (t == 'hs'):
        return Activation('hard_sigmoid',name='hard_sigmoid_'+str(ind))(x)
    if (t == 's'):
        return Activation('sigmoid',name='sigmoid_'+str(ind))(x)
    if (t == 't'):
        return Activation('tanh',name='tanh_'+str(ind))(x)
    if (t == 'm'):
        return mish(x)

    return x

def atrous_block(x_i,ind=0,nf=32,fs=3,strides=1,act='l',dropout_rate=None,weight_decay=5e-4,pool=0,FUS = 'max'):
    ki = initializers.he_normal()
    kr = regularizers.l2(weight_decay)
    x=[]
    d=[]
    ab=3
    
    redu_r = np.shape(x_i)[-1] // 2
    if(ind>0):
        x_i = Conv2D(redu_r, (1, 1), strides=(1, 1), kernel_initializer=ki, kernel_regularizer=kr, padding='same', use_bias=False,name='conv_2d_redu_' + str(ind))(x_i)
        x_i = BatchNormalization(name='atrous_redu_bn_' + str(ind))(x_i)
        x_i = activation(x_i, 'atrous_redu_act_'+str(ind), act)

    def mininet(x,dr,ind):
        m = DepthwiseConv2D(kernel_size=fs, kernel_initializer=ki, kernel_regularizer=kr, strides=strides, padding='same', use_bias=False,dilation_rate=dr + 1, name='atrous_depth_conv_' + str(ind) + '_' + str(dr))(x)
        m = BatchNormalization(name='atrous_bn_'+ str(ind) + '_' + str(dr))(m)
        m = activation(m, 'atrous_act'+str(ind)+'_'+str(dr), act)
       
        return m

    for i in range(ab):
        x.append(
            mininet(x_i,i,ind)
        )
        d.append(x[i].shape[1])

    mr=[x_i]

    for i in range(0,len(d)):
        if(d[0]==d[i]):
            mr.append(x[i])

    if(len(mr) > 1):
        f = fusion_block(mr, str(ind), FUS)
    else:
        f=x[0]

    b = Conv2D(nf, (1, 1), strides=(1, 1), kernel_initializer=ki, kernel_regularizer=kr, padding='same', use_bias=False,name='conv_2d_' + str(ind))(f)
    b = BatchNormalization(name='cnv_bn_'+ str(ind))(b)
    b = activation(b, 'ccnv_act'+str(ind), act)


    if dropout_rate != None and dropout_rate != 0.:
       b = Dropout(dropout_rate)(b)

    return b

def make_ACFF_model(H,W,C, fus='add'):

    input_shape = [H, W, 3]
    inp = Input(shape=input_shape)
    
    x = inp
    act = 'm'

    wd = 5e-4
    x = Conv2D(32, (5,5), name= 'convI',strides=2,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = activation(x, 'ccnv_act0', act)
    x = atrous_block(x,ind=1,nf=64,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=2,nf=96,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=3,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=4,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = atrous_block(x,ind=5,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = atrous_block(x,ind=6,nf=256,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)

    x = separable_conv_block(x, C, kernel_size=1, stride=1, weight_decay=wd, act=act,dropout_rate=0.)

    x = GlobalAveragePooling2D()(x)
    cls = Activation('softmax', name='class_branch')(x)
    
    return inp, cls


def train():

    K.clear_session()

    train_data_dir=args['data_dir']
    val_data_dir=args['data_dir']

    inp,cls = make_ACFF_model(img_height,img_width,C=num_classes, fus='avg')
    model = Model(inputs=[inp], outputs=[cls])
    model.summary()

    seed = 22
    rnd.seed(seed)
    np.random.seed(seed)

    dsplit = 0.2

    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=properties.rotation_range,  
        width_shift_range=properties.width_shift_range,  
        height_shift_range=properties.height_shift_range,  
        shear_range=properties.shear_range,  
        zoom_range=properties.zoom_range,  
        horizontal_flip=properties.should_hflip,  
        fill_mode=properties.fill_mode,  
        validation_split=dsplit)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
        )

    validation_datagen = ImageDataGenerator(rescale=1./255.,
        preprocessing_function = None,validation_split=dsplit)

    validation_generator = validation_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
        )

    print(f'./weights/model-{uuid.uuid4()}-en.h5', ' is the checkpoint')
    checkpoint = ModelCheckpoint(f'./weights/model-{uuid.uuid4()}-en.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=False)
    weight_checkpoint = ModelCheckpoint(f'./weights/model-weights-{uuid.uuid4()}-en.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True)

    opt = tf.keras.optimizers.Adam(lr=lr_init)
    cd = cosine_decay(epochs_tot=epochs,initial_lrate=lr_init,period=1,fade_factor=1.,min_lr=1e-3)

    lrs = LearningRateScheduler(cd,verbose=1)

    lrr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=10, min_lr=1e-6,verbose=1)

    callbacks_list = [lrs,checkpoint,weight_checkpoint]


    SMOOTHING=0.1
    loss = CategoricalCrossentropy(label_smoothing=SMOOTHING)

    model.compile(optimizer=opt,metrics=keras.metrics.CategoricalAccuracy(),loss=loss)

    model.fit(x=train_generator,\
                    steps_per_epoch=train_generator.samples // batch_size,epochs=epochs,\
                    verbose=1,validation_data=validation_generator,validation_steps = validation_generator.samples // batch_size,callbacks=callbacks_list,workers=num_workers, class_weight={0: 0.7258164777889351, 1: 2.1950831525668835, 2: 0.8571347902196623})

def validation(img, fusion):
    img_height,img_width = 240, 240
    num_classes = properties.num_classes
    inp,cls = make_ACFF_model(img_height,img_width, C=num_classes, fus=fusion)
    model = Model(inputs=[inp], outputs=[cls])
    if fusion == 'add':
        weights_path = 'model_add_fus.h5'
    elif fusion == 'max':
        weights_path = 'model_max_fus.h5'
    elif fusion == '':
        weights_path = ''
    elif fusion == '':
        weights_path = ''
    model.load_weights(weights_path)
    return model.predict(img).argmax(axis=1)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-d', '--data_dir', required=True, help='path to input dataset'
    )
    args = vars(ap.parse_args())
    train()

