## Task 2-1: 
#          Examine the influence of parameter initializers on the following networks:
#                        1. fully connected network
#                        2. Locally connected network
#                        3. convolution network
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import pandas as pd

from full_connected_net import My_network as full_net
from locally_connected_net import My_network as local_net
from convolution_net import My_network as conv_net

## initilializer list
inititalizers_list = ['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'variance_scaling']
tot1 = len(inititalizers_list)

## load data
train_data = np.loadtxt('zip_train.txt')
# print('train size', train_data.shape)
Y_train = train_data[:,0]
X_train_full = train_data[:,1:257]
X_train_graph = np.reshape(train_data[:,1:257],(train_data.shape[0],16,16,1))

test_data = np.loadtxt('zip_test.txt')
# print('test size', test_data.shape)
Y_test = test_data[:,0]
X_test_full = test_data[:,1:257]
X_test_graph = np.reshape(test_data[:,1:257],(test_data.shape[0],16,16,1))

# hyper-parameters
CLASSES_SIZE = 10   # number of classes for MNIST
LR = 0.1            # learning rate
my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)

epoch_size = 10000
batch_size = 32

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)


############# 1. fully-connected network ##################
# create a void table
Converge_epoch_full = pd.DataFrame(columns = inititalizers_list)
Converge_epoch_full.insert(0,'initializer_name',inititalizers_list) 
Converge_epoch_full = Converge_epoch_full.set_index('initializer_name')

# test
for idx1 in range(tot1):
    for idx2 in range(tot1):
        net = full_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size,kernel_init = inititalizers_list[idx1], bias_init = inititalizers_list[idx2])
        reg_net = net.define_network(input_size = 256, output_size = CLASSES_SIZE)
        Loss, Train_error, Test_error, Converge_epoch_full.loc[inititalizers_list[idx1],inititalizers_list[idx2]] = net.train_loop(X_train_full,Y_train,X_test_full,Y_test)

# result
Converge_epoch_full.to_csv('Task2_Initilizer_full.csv')
print('testing learning speed by using different initializer (fully connected network)', Converge_epoch_full)


############# 2. locally-connected network ##################
# create a void table
Converge_epoch_local = pd.DataFrame(columns = inititalizers_list)
Converge_epoch_local.insert(0,'initializer_name',inititalizers_list) 
Converge_epoch_local = Converge_epoch_local.set_index('initializer_name')

# test
for idx1 in range(tot1):
    for idx2 in range(tot1):
        net = local_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size,kernel_init = inititalizers_list[idx1], bias_init = inititalizers_list[idx2])
        reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)
        Loss, Train_error, Test_error, Converge_epoch_local.loc[inititalizers_list[idx1],inititalizers_list[idx2]] = net.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)

# Result
Converge_epoch_local.to_csv('Task2_Initilizer_locally_connected.csv')
print('testing learning speed by using different initializer (lcoally connected network)', Converge_epoch_local)


############# 3. concolutional network ##################
# create a void table
Converge_epoch_conv = pd.DataFrame(columns = inititalizers_list)
Converge_epoch_conv.insert(0,'initializer_name',inititalizers_list) 
Converge_epoch_conv = Converge_epoch_conv.set_index('initializer_name')

# test
for idx1 in range(tot1):
    for idx2 in range(tot1):
        net = conv_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size,kernel_init = inititalizers_list[idx1], bias_init = inititalizers_list[idx2])
        reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)
        Loss, Train_error, Test_error, Converge_epoch_conv.loc[inititalizers_list[idx1],inititalizers_list[idx2]] = net.train_loop(X_train_graph,Y_train,X_test_graph,Y_test)

# Result
Converge_epoch_conv.to_csv('Task2_Initilizer_conv.csv')
print('testing learning speed by using different initializer (concolutional network)', Converge_epoch_conv)