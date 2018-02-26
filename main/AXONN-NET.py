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
import matplotlib.pyplot as plt

# ignore this
# this is for the Tensorflow settings. Usually the default settings are the fastest
#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.Session(config=config)
#K.set_session(session)


def generate(batch_size,train):
    if train:
        index = 0;
    else:
        index = 8000/batch_size
    while 1:
        data = tables.open_file('/media/lennart/Volume/Downloads/10000images.hdf5',mode='r')
        images = np.array(data.root.images[index*batch_size:(index+1)*batch_size])
        labels = np.array(data.root.pathologies[index*batch_size:(index+1)*batch_size])
        images = images.astype('float32')
        images *= 1/255
        images = images.reshape(images.shape[0], 1024,1024,1)
        images = np.tile(images,(1,1,1,3))

        labels = labels.astype('float32')
        index +=1
        data.close()

        if train:
            if index*batch_size >= (8000-batch_size):
                index = 0
        else:
            if index*batch_size >= (10000-batch_size):
                index = 8000/batch_size
        yield images, labels


#defining batch size, epochs and output num
batch_size = 3
num_classes = 14
epochs = 3

# input image dimensions
img_rows, img_cols = 1024, 1024


#y_train = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',0,800)
#y_test = keras.utils.HDF5Matrix('./1000_data.hdf5', 'image_labels',800,1000)

'''
conv_base = VGG16(weights='imagenet', include_top=False, input_shape = (img_rows,img_cols,3))


for layer in conv_base.layers[:-4]:
	layer.trainable = False
for layer in conv_base.layers:
	print(layer, layer.trainable)
'''
#add some new layers to the pretrained network to get the desired results
model = Sequential()
#model.add(conv_base)
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'same',input_shape=(img_rows,img_cols,3)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding = 'same',input_shape=(img_rows,img_cols,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
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
training_gen = generate(batch_size,True)
valid_gen = generate(batch_size,False)

#fit the model 

history = model.fit_generator(generator=training_gen, verbose = 1, epochs = epochs, steps_per_epoch = 8000/batch_size, validation_data = valid_gen, validation_steps = 2000/batch_size)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
model.save('./chest_xray_net.h5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
