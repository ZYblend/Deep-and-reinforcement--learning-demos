## Ensemble learning
# model 1: fully connected network
# model 2: locally connected network
# model 3: convolution network

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils

from full_connected_net import My_network as full_net
from locally_connected_net import My_network as local_net
from convolution_net import My_network as conv_net

## load data
train_data = np.loadtxt('zip_train.txt')
print('train size', train_data.shape)
Y_train = train_data[:,0]
X_train_full = train_data[:,1:257]
X_train_conv = np.reshape(train_data[:,1:257],(train_data.shape[0],16,16,1))

test_data = np.loadtxt('zip_test.txt')
print('test size', test_data.shape)
Y_test = test_data[:,0]
X_test_full = test_data[:,1:257]
X_test_conv = np.reshape(test_data[:,1:257],(test_data.shape[0],16,16,1))

# hyper-parameters
CLASSES_SIZE = 10   # number of classes for MNIST
LR = 0.01   # learning rate
my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)

epoch_size = 10000
batch_size = 32

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)


## ensemble learning

# define fully connected network
# net1 = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
# model1 = net1.define_network(input_size = 256, output_size = CLASSES_SIZE)
model1 = keras.models.load_model('my_model_fullconnected_forEnsemble.h5')

# define locally connected network
# net2 = local_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
# model2 = net2.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)
model2 = keras.models.load_model('my_model_local_connectted_forEnsemble.h5')

# define convolution network
# net3 = conv_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
# model3 = net3.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)
model3 = keras.models.load_model('my_model_conv_forEnsemble.h5')

# average ensemble
y1_pred = model1(X_test_full)
m1      = tf.keras.metrics.CategoricalAccuracy()
m1.update_state(Y_test,y1_pred)
acc1    = m1.result().numpy()

y2_pred = model2(X_test_conv)
m2      = tf.keras.metrics.CategoricalAccuracy()
m2.update_state(Y_test,y2_pred)
acc2    = m2.result().numpy()

y3_pred = model3(X_test_conv)
m3      = tf.keras.metrics.CategoricalAccuracy()
m3.update_state(Y_test,y3_pred)
acc3    = m3.result().numpy()

weights = [1/3,1/3,1/3]
y_pred = weights[0] * y1_pred + weights[1] * y2_pred +  weights[2] * y3_pred
m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(Y_test,y_pred)
acc = m.result().numpy()

print('fully connected testing accuracy:', acc1)
print('lcoally connected testing accuracy:', acc2)
print('conv testing accuracy:', acc3)
print('Ensemblly testing accuracy:', acc)
