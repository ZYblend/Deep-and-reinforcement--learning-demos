''' Extra credits: adversrial training on fully connected network
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

from full_connected_net import My_network as full_net

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
adv_epoch_size = 10000
batch_size = 32

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)


############### Without adversiral training ##############
# define network
net = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size, adv_EPOCH_SIZE=adv_epoch_size)
reg_net = net.define_network(input_size = 256, output_size = CLASSES_SIZE)

# Train network
Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train_full,Y_train,X_test_full,Y_test)

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
epoch_list = np.linspace(0,epoch_size,epoch_size)
fig, ax = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax[0].plot(epoch_list,Loss, color = 'b')
ax[0].set_title('Loss curve')
ax[0].set_xlabel('Epochs')
ax[1].plot(epoch_list,Train_error, color = 'b')
ax[1].set_title('Training Accuracy')
ax[1].set_xlabel('Epochs')
ax[2].plot(epoch_list,Test_error, color = 'b')
ax[2].set_title('Testing Accuracy')
ax[2].set_xlabel('Epochs')


############### With adversiral training ##############
Loss_adv, Train_error_adv, Test_error_adv = net.adversarial_training(X_train_full,Y_train,X_test_full,Y_test)

# save network
reg_net.save('my_model_fullconnected_adv.h5')

# evaluation (accuracy api)
Y_train_pred = reg_net(X_train_full)
m3 = tf.keras.metrics.CategoricalAccuracy()
train_acc = m3.update_state(Y_train,Y_train_pred)
print('Train set accuracy (with advasarial training):',m3.result().numpy())

Y_test_pred = reg_net(X_test_full)
m4 = tf.keras.metrics.CategoricalAccuracy()
test_acc = m4.update_state(Y_test,Y_test_pred)
print('Test set accuracy (with advasarial training):', m4.result().numpy())

# plotting
adv_epoch_list = np.linspace(epoch_size,epoch_size + adv_epoch_size,adv_epoch_size)
ax[0].plot(adv_epoch_list,Loss_adv,color='r')
ax[1].plot(adv_epoch_list,Train_error_adv,color='r')
ax[2].plot(adv_epoch_list,Test_error_adv,color='r')
fig.suptitle('blue: normal training, red: adversrial training')
plt.legend()
plt.show()
