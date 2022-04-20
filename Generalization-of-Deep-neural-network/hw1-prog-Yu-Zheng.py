import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# hyper-parameters
CLASSES_SIZE = 10   # number of classes for MNIST
EPOCH_SIZE = 200
BATCH_SIZE = 256

LR = 0.001   # learning rate

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# randomize the training classes
y_train = tf.transpose(tf.random.categorical(tf.constant([[1.,1.,1.,1.,1.,1.,1.,1.,1.]]), y_train.shape[0]))


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, CLASSES_SIZE)

Y_test = np_utils.to_categorical(y_test, CLASSES_SIZE)
#layers.Dropout(.2),
# define a simple multi-classes classfication network
multiclass_classfier = tf.keras.Sequential([
                                            layers.Dense(512, activation='relu', kernel_initializer='he_uniform', input_dim=784),                                            
                                            layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
                                            layers.Dense(CLASSES_SIZE, activation='softmax')
                                            ])
    
multiclass_classfier.summary()

# define compiler
multiclass_classfier.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=LR),
                  metrics=['accuracy'])

# train
history = multiclass_classfier.fit(X_train, Y_train,
                                   batch_size=BATCH_SIZE,
                                   epochs=EPOCH_SIZE,
                                   verbose=1,
                                   validation_data=(X_test, Y_test))

# evaluation
fig,ax = plt.subplots(1,2)
ax[0].plot(history.epoch, history.history['loss'])
ax[0].set_title('Loss')
ax[1].plot(history.epoch, history.history['accuracy'])
ax[1].set_title('Accuracy')

plt.legend()
plt.show()

score_train_set = multiclass_classfier.evaluate(X_train, Y_train, verbose=0)
print('Train set accuracy:', score_train_set[1])

score_test_set = multiclass_classfier.evaluate(X_test, Y_test, verbose=0)
print('Test set accuracy:', score_test_set[1])