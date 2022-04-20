## Task 2-2: 
#          Examine the influence of learning rate on the following networks:
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


###################### 1. fully connected network #######################
LR_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]   # learning rate
# LR_list = [0.15, 0.2, 0.3, 0.4, 0.5]

fig1, ax1 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
labels = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
# labels = ['0.15', '0.2', '0.3', '0.4', '0.5']

for iter in range(len(LR_list)):
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR_list[iter])
    # define network
    net = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
    reg_net = net.define_network(input_size = 256, output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train_full,Y_train,X_test_full,Y_test)
    
    ax1[0].plot(Loss,label=labels[iter])
    ax1[0].set_title('Loss curve')
    ax1[0].set_xlabel('Epochs')
    ax1[0].legend()
    ax1[1].plot(Train_error,label=labels[iter])
    ax1[1].set_title('Training Accuracy')
    ax1[1].set_xlabel('Epochs')
    ax1[1].legend()
    ax1[2].plot(Test_error,label=labels[iter])
    ax1[2].set_title('Testing Accuracy')
    ax1[2].set_xlabel('Epochs')
    ax1[2].legend()
fig1.suptitle('fully-connected network')
plt.savefig("learning_rate1_fully.png")

############# 2. locally-connected network ##################
LR_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]   # learning rate
# LR_list = [ 0.2, 0.6, 1.0, 1.4, 1.8, 2]
fig2, ax2 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
labels = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
# labels = [ '0.2', '0.6', '1.0', '1.4','1.8','2']

for iter in range(len(LR_list)):
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR_list[iter])
    # define network
    net = local_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
    reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)
    
    ax2[0].plot(Loss,label=labels[iter])
    ax2[0].set_title('Loss curve')
    ax2[0].set_xlabel('Epochs')
    ax2[0].legend()
    ax2[1].plot(Train_error,label=labels[iter])
    ax2[1].set_title('Training Accuracy')
    ax2[1].set_xlabel('Epochs')
    ax2[1].legend()
    ax2[2].plot(Test_error,label=labels[iter])
    ax2[2].set_title('Testing Accuracy')
    ax2[2].set_xlabel('Epochs')
    ax2[2].legend()
fig2.suptitle('Lcoally-connected network')
plt.savefig("learning_rate2_local.png")

##################### 2. Convolutional network #####################
LR_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]   # learning rate
# LR_list = [0.15, 0.2, 0.3, 0.4, 0.5]

fig3, ax3 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
labels = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
# labels = ['0.15', '0.2', '0.3', '0.4', '0.5']

for iter in range(len(LR_list)):
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR_list[iter])
    # define network
    net = conv_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
    reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)
    
    ax3[0].plot(Loss,label=labels[iter])
    ax3[0].set_title('Loss curve')
    ax3[0].set_xlabel('Epochs')
    ax3[0].legend()
    ax3[1].plot(Train_error,label=labels[iter])
    ax3[1].set_title('Training Accuracy')
    ax3[1].set_xlabel('Epochs')
    ax3[1].legend()
    ax3[2].plot(Test_error,label=labels[iter])
    ax3[2].set_title('Testing Accuracy')
    ax3[2].set_xlabel('Epochs')
    ax3[2].legend()

fig3.suptitle('Convolution network')
plt.savefig("learning_rate1_conv.png")
plt.show()