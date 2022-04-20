# Task3: Drop rate
# when drop rate is higher than 0.8, the network does not converge

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from full_connected_net import My_network as full_net

## load data
train_data = np.loadtxt('zip_train.txt')
print('train size', train_data.shape)
Y_train = train_data[:,0]
X_train = train_data[:,1:257]

test_data = np.loadtxt('zip_test.txt')
print('test size', test_data.shape)
Y_test = test_data[:,0]
X_test = test_data[:,1:257]


CLASSES_SIZE = 10   # number of classes for MNIST
LR = 0.1   # learning rate
kernel = 'random_normal'
bias = 'zeros'

epoch_size = 100000
batch_size = 64
my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)

# without dropout
net = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size, kernel_init = kernel, bias_init = bias)
reg_net = net.define_network(input_size = 256, output_size = CLASSES_SIZE)
# Train network
Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train,Y_train,X_test,Y_test)
# evaluation (accuracy api)
train_acc_nom = net.accuracy_metric(X_train,Y_train,reg_net)
test_acc_nom  = net.accuracy_metric(X_test,Y_test,reg_net)


## Dropout Layer
rate_list = [0.1, 0.2,  0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
labels = ['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
#rate_list = [0.3,  0.4]
#labels = ['0.3', '0.4']

## testing dropout
Train_acc = np.zeros((len(rate_list),1))
Test_acc = np.zeros((len(rate_list),1))
for iter in range(len(rate_list)):
    # define network
    net = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size, kernel_init = kernel, bias_init = bias)
    reg_net1 = net.define_network_droplayer(input_size = 256, output_size = CLASSES_SIZE, drop_rate=rate_list[iter])
    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train,Y_train,X_test,Y_test)

    # evaluation (accuracy)
    Train_acc[iter] = net.accuracy_metric(X_train,Y_train,reg_net1)
    Test_acc[iter]  = net.accuracy_metric(X_test,Y_test,reg_net1)

## plotting
np.savetxt('train_acc_dropout.txt',Train_acc)
np.savetxt('test_acc_dropout.txt',Test_acc)

#Train_acc = np.loadtxt('train_acc_dropout.txt')
#Test_acc = np.loadtxt('test_acc_dropout.txt')
#rate_list = [0, 0.1, 0.2,  0.3,  0.4, 0.5]
fig, ax = plt.subplots(1,2)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax[0].scatter(rate_list, Train_acc)
ax[0].set_xlabel('Drop out Rate')
ax[0].set_xticks(rate_list)
ax[0].set_ylabel('Training Accuracy')
ax[0].axhline(y=train_acc_nom,xmin=rate_list[0],xmax=rate_list[len(rate_list)-1],c="red",linewidth=0.5,zorder=0)
ax[1].scatter(rate_list, Test_acc)
ax[1].set_xlabel('Drop out Rate')
ax[1].set_xticks(rate_list)
ax[1].set_ylabel('Testing Accuracy')
ax[1].axhline(y=test_acc_nom,xmin=rate_list[0],xmax=rate_list[len(rate_list)-1],c="red",linewidth=0.5,zorder=0)
plt.savefig("average_drop_rate_fully.png")
plt.show()

