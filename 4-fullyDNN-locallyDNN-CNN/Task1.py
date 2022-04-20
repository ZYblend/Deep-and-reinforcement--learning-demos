## Task 1 : 
#          Test the following networks on the digit handwriting task:
#                        1. fully connected network
#                        2. Locally connected network
#                        3. convolution network

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils

from full_connected_net import My_network as full_net
from locally_connected_net import My_network as local_net
from convolution_net import My_network as conv_net

# import dataset 
train_data = np.loadtxt('zip_train.txt')
print('train size', train_data.shape)
Y_train = train_data[:,0]
X_train_full = train_data[:,1:257]
X_train_graph = np.reshape(train_data[:,1:257],(train_data.shape[0],16,16,1))

test_data = np.loadtxt('zip_test.txt')
print('test size', test_data.shape)
Y_test = test_data[:,0]
X_test_full = test_data[:,1:257]
X_test_graph = np.reshape(test_data[:,1:257],(test_data.shape[0],16,16,1))

# hyper-parameters
CLASSES_SIZE = 10   # number of classes for MNIST
LR = 0.1   # learning rate
my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)

epoch_size = 10000
batch_size = 32

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)

# network testing
############### 1. fully connected network ##############
# define network
net1 = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
reg_net = net1.define_network(input_size = 256, output_size = CLASSES_SIZE)

# Train network
Loss, Train_error, Test_error, converge_epoch = net1.train_loop(X_train_full,Y_train,X_test_full,Y_test)

# save network
reg_net.save('my_model_fullconnected.h5')

# evaluation (accuracy api)
Y_train_pred = reg_net(X_train_full)
m1 = tf.keras.metrics.CategoricalAccuracy()
train_acc = m1.update_state(Y_train,Y_train_pred)
print('Train set accuracy:',m1.result().numpy())

Y_test_pred = reg_net(X_test_full)
m2 = tf.keras.metrics.CategoricalAccuracy()
test_acc = m2.update_state(Y_test,Y_test_pred)
print('Test set accuracy:', m2.result().numpy())

# convergency speed
print('loss converge when at epoch',converge_epoch)

# plotting
fig1, ax1 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax1[0].plot(Loss)
ax1[0].set_title('Loss curve')
ax1[0].set_xlabel('Epochs')
ax1[1].plot(Train_error)
ax1[1].set_title('Training Accuracy')
ax1[1].set_xlabel('Epochs')
ax1[2].plot(Test_error)
ax1[2].set_title('Testing Accuracy')
ax1[2].set_xlabel('Epochs')
fig1.suptitle('fully-connected network')
plt.savefig("Curve_fully_connected_"+str(epoch_size)+".png")
# plt.show()


##################### 2. locally connected network #####################
# define network
net2 = local_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
reg_net = net2.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

# Train network
Loss, Train_error, Test_error, converge_epoch = net2.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)

# save network
reg_net.save('my_model_local_connectted.h5')

# evaluation (accuracy)
Y_train_pred = reg_net(X_train_graph)
m1 = tf.keras.metrics.CategoricalAccuracy()
train_acc = m1.update_state(Y_train,Y_train_pred)
print('Train set accuracy:',m1.result().numpy())

Y_test_pred = reg_net(X_test_graph)
m2 = tf.keras.metrics.CategoricalAccuracy()
test_acc = m2.update_state(Y_test,Y_test_pred)
print('Test set accuracy:', m2.result().numpy())


# plotting
fig2, ax2 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax2[0].plot(Loss)
ax2[0].set_title('Loss curve')
ax2[0].set_xlabel('Epochs')
ax2[1].plot(Train_error)
ax2[1].set_title('Training Accuracy')
ax2[1].set_xlabel('Epochs')
ax2[2].plot(Test_error)
ax2[2].set_title('Testing Accuracy')
ax2[2].set_xlabel('Epochs')
fig2.suptitle('Lcoally-connected network')
plt.savefig("Curve_locally_connected_"+str(epoch_size)+".png")

# plt.show()

##################### 3. Convolutional network #####################
# # define network
net3 = conv_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
reg_net = net3.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

# # Train network
Loss, Train_error, Test_error, converge_epoch = net3.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)

# save network
reg_net.save('my_model_conv.h5')

# evaluation (accuracy)
Y_train_pred = reg_net(X_train_graph)
m1 = tf.keras.metrics.CategoricalAccuracy()
train_acc = m1.update_state(Y_train,Y_train_pred)
print('Train set accuracy:',m1.result().numpy())

Y_test_pred = reg_net(X_test_graph)
m2 = tf.keras.metrics.CategoricalAccuracy()
test_acc = m2.update_state(Y_test,Y_test_pred)
print('Test set accuracy:', m2.result().numpy())

# plotting
fig3, ax3 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax3[0].plot(Loss)
ax3[0].set_title('Loss curve')
ax3[0].set_xlabel('Epochs')
ax3[1].plot(Train_error)
ax3[1].set_title('Training Accuracy')
ax3[1].set_xlabel('Epochs')
ax3[2].plot(Test_error)
ax3[2].set_title('Testing Accuracy')
ax3[2].set_xlabel('Epochs')
fig3.suptitle('Convolution network')
plt.savefig("Curve_convolution_"+str(epoch_size)+".png")

plt.show()


