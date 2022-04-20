# Task2 -- 3:
# Influence of the Batch Size with Batchnormalization in conv
# Tried different values following 2*k, 32 always performs the best    

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from convolution_net import My_network as conv_net

## load data
train_data = np.loadtxt('zip_train.txt')
print('train size', train_data.shape)
Y_train = train_data[:,0]
X_train = np.reshape(train_data[:,1:257],(train_data.shape[0],16,16,1))

test_data = np.loadtxt('zip_test.txt')
print('test size', test_data.shape)
Y_test = test_data[:,0]
X_test = np.reshape(test_data[:,1:257],(test_data.shape[0],16,16,1))
print(X_train.shape)


CLASSES_SIZE = 10   # number of classes for MNIST
LR = 0.1   # learning rate
kernel = 'random_normal'
bias = 'zeros'

epoch_size = 10000
batch_size = 32

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)


############# 3. Batch Size ################
BS_list = [4, 8, 16, 32, 64, 128]   # batch size
labels = ['4', '8', '16', '32', '64', '128']

# fig, ax = plt.subplots(1,3,figsize=(12,4))
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
    
for iter in range(len(BS_list)):
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)
    # define network
    net = conv_net(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size, kernel_init = kernel, bias_init = bias)
    reg_net = net.define_network_BN(input_size = (16,16,1), output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train,Y_train,X_test,Y_test)
    
    print("converge epoch:", converge_epoch)
    # np.savetxt('loss_conv_'+labels[iter]+'.txt',Loss)
    # np.savetxt('trainerr_conv_'+labels[iter]+'.txt',Train_error)
    # np.savetxt('testerr_conv_'+labels[iter]+'.txt',Test_error)

#     ax[0].plot(Loss,label=labels[iter])
#     ax[0].set_title('Loss curve')
#     ax[0].set_xlabel('Epochs')
#     ax[0].legend()
#     ax[1].plot(Train_error,label=labels[iter])
#     ax[1].set_title('Training Accuracy')
#     ax[1].set_xlabel('Epochs')
#     ax[1].legend()
#     ax[2].plot(Test_error,label=labels[iter])
#     ax[2].set_title('Testing Accuracy')
#     ax[2].set_xlabel('Epochs')
#     ax[2].legend()

# plt.savefig("batch_size_conv.png")
# plt.show()

