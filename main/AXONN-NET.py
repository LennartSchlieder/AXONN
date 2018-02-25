from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras import backend as K
from keras.applications import VGG16
import numpy as np
import tables

# ignore this
# this is for the Tensorflow settings. Usually the default settings are the fastest
#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.Session(config=config)
#K.set_session(session)


def generate(batch_size):
    index = 0;
    while 1:
        data = tables.open_file('./1000_data.hdf5',mode='r')
        images = np.array(data.root.images[index*batch_size:(index+1)*batch_size])
        labels = np.array(data.root.pathologies[index*batch_size:(index+1)*batch_size])
        images = images.astype('float32')
        images *= 1/255
        images = images.reshape(images.shape[0], 1024,1024,1)
        images = np.tile(images,(1,1,1,3))

        labels = labels.astype('float32')
        index +=1
        data.close()

      
        if index*batch_size > 1000:
            index = 0
        yield images, labels


#defining batch size, epochs and output num
batch_size = 5
num_classes = 14
epochs = 1

# input image dimensions
img_rows, img_cols = 1024, 1024


#y_train = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',0,800)
#y_test = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',800,1000)


conv_base = VGG16(weights='imagenet', include_top=False, input_shape = (img_rows,img_cols,3))


for layer in conv_base.layers[:-4]:
	layer.trainable = False
for layer in conv_base.layers:
	print(layer, layer.trainable)

#add some new layers to the pretrained network to get the desired results
model = Sequential()
model.add(conv_base)

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

#model.add(Flatten(input_shape = input_shape))
#model.add(Dense(num_classes, activation='linear', use_bias = False))

#sdg = optimizers.SGD(lr=0.2, momentum=0.01, decay=0.01, nesterov=False)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

# chose categorical_crossentropy to avoid slow learning for the sigma derivative. 
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#Create the generators for the test and train batches
training_gen = generate(batch_size)
valid_gen = generate(batch_size)

#fit the model 

model.fit_generator(generator=training_gen, verbose = 1, epochs = epochs, steps_per_epoch = 1000/batch_size, validation_data = valid_gen, validation_steps = 1)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
model.save('./chest_xray_net.h5')
Conv2D, MaxPooling2D
from keras import optimizers
from keras import backend as K
from keras.applications import VGG16
import numpy as np
import tables

# ignore this
# this is for the Tensorflow settings. Usually the default settings are the fastest
#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.Session(config=config)
#K.set_session(session)


def generate(batch_size):
    index = 0;
    while 1:
        data = tables.open_file('./1000_data.hdf5',mode='r')
        images = np.array(data.root.images[index*batch_size:(index+1)*batch_size])
        labels = np.array(data.root.pathologies[index*batch_size:(index+1)*batch_size])
        images = images.astype('float32')
        images *= 1/255
        images = images.reshape(images.shape[0], 1024,1024,1)
        images = np.tile(images,(1,1,1,3))
        #print(images.shape)
        #print(labels)
        labels = labels.astype('float32')
        index +=1
        data.close()
        #images = np.zeros((batch_size,1024,1024,3))
        #labels = np.zeros((batch_size,14))
       # print(index)
        if index*batch_size > 1000:
            index = 0
        yield images, labels


#defining batch size, epochs and output num
batch_size = 5
num_classes = 14
epochs = 1

# input image dimensions
img_rows, img_cols = 1024, 1024

# load the data
#table = tables.open_file('./1000_data.hdf5',mode='r')
#x_train = keras.utils.HDF5Matrix('./1000_data.hdf5', 'images',0,800,normalizer = normalize)
#x_test = keras.utils.HDF5Matrix('./1000_data.hdf5', 'images',800,1000,normalizer = normalize)

#y_train = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',0,800)
#y_test = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',800,1000)

#print(x_train.shape)

#x_train = x_train.reshape(x_train.shape[0], 1, img_rows,img_cols)
#x_train = np.tile(x_train,(1,3,1,1))

#x_test = x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
#x_test = np.tile(x_test,(1,3,1,1))

#print(x_train.shape)
'''
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows*2, img_cols*2)
    x_train = np.tile(x_train,(1,3,2,2))

    #x_train = x_train[0:10]
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows*2, img_cols*2)
    x_test = np.tile(x_test,(1,3,2,2))


    #x_test = x_test[0:10]
    input_shape = (3, img_rows*2, img_cols*2)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = np.tile(x_train,(1,2,2,3))

    #x_train = x_train[0:10]
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = np.tile(x_test,(1,2,2,3))
    #x_test = x_test[0:10]
    input_shape = (img_rows*2, img_cols*2, 3)
'''
#print(y_train.shape)
#print(x_train.shape)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

#y_train = keras.utils.to_categorical(y_train,num_classes)
#y_test = keras.utils.to_categorical(y_test,num_classes)


#loading a pretrained convolutional part of a network
#conv_base = VGG16(weights='imagenet', include_top=False, input_shape = (1024,1024,3))
conv_base = VGG16(weights='imagenet', include_top=False, input_shape = (img_rows,img_cols,3))


for layer in conv_base.layers[:-4]:
	layer.trainable = False
for layer in conv_base.layers:
	print(layer, layer.trainable)

#add some new layers to the pretrained network to get the desired results
model = Sequential()
model.add(conv_base)
#model.add(Conv2D(32, 3, 3,
#                       border_mode='same',
#                       input_shape=(img_rows,img_cols,3)))
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'same',input_shape=(img_rows,img_cols,3)))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

#model.add(Flatten(input_shape = input_shape))
#model.add(Dense(num_classes, activation='linear', use_bias = False))

#sdg = optimizers.SGD(lr=0.2, momentum=0.01, decay=0.01, nesterov=False)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

training_gen = generate(batch_size)
valid_gen = generate(batch_size)

images = np.zeros((batch_size,1024,1024,3))
labels = np.zeros((batch_size,14))
images_test = np.zeros((batch_size,1024,1024,3))
labels_test = np.zeros((batch_size,14))
#model.fit(images, labels,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(images_test, labels_test))

model.fit_generator(generator=training_gen, verbose = 1, epochs = epochs, steps_per_epoch = 1000/batch_size, validation_data = valid_gen, validation_steps = 1)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
model.save('./chest_xray_net.h5')

