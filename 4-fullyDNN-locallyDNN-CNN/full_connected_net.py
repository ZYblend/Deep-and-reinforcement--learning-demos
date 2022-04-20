import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import pandas as pd

class My_network(object):
    def __init__(self,optimizer, EPOCH_SIZE=300, BATCH_SIZE=36, kernel_init = 'random_normal', bias_init = 'zeros', adv_EPOCH_SIZE = 300):
        # hyper-parameters
        self.EPOCH_SIZE = EPOCH_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.adv_EPOCH_SIZE = adv_EPOCH_SIZE

        self.optimizer = optimizer

        self.kernel_init = kernel_init
        self.bias_init = bias_init

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.m1 = tf.keras.metrics.CategoricalAccuracy()
        self.m2 = tf.keras.metrics.CategoricalAccuracy()

        # for covergency speed
        self.count = 0
        self.tol_counts = 100
        self.converge_epoch = 0

        # for adversarial training
        self.epsilon = 0.1
    
    def accuracy_metric(self,x,y,network):
        ''' This has been tested, it is same as the  tf.keras.metrics.CategoricalAccuracy() API
        '''
        y_pred = network(x).numpy()

        # the maximum idx corresponds to the category id
        for row in range(y_pred.shape[0]):
            idx = np.argmax(y_pred[row,:])
            y_pred[row,:] = 0
            y_pred[row,idx] = 1
        
        false_sample = np.sum(np.absolute(y_pred-y))/2
        total_sample = y_pred.shape[0]
        
        return 1-(false_sample/total_sample)

    
    def define_network(self,input_size,output_size):
        self.reg_net = tf.keras.Sequential([
                                        layers.Dense(512, activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, input_dim=input_size),                                            
                                        layers.Dense(512, activation='relu',kernel_initializer=self.kernel_init,bias_initializer=self.bias_init),
                                        layers.Dense(256, activation='tanh',kernel_initializer=self.kernel_init,bias_initializer=self.bias_init),
                                        layers.Dense(output_size, activation='softmax')
                                        ])
        self.reg_net.summary()
        return self.reg_net

    def define_network_droplayer(self,input_size,output_size, drop_rate):
        self.reg_net = tf.keras.Sequential([
                                        layers.Dense(512, activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, input_dim=input_size),                                            
                                        # layers.BatchNormalization(),
                                        # layers.Dropout(drop_rate),
                                        layers.Dense(512, activation='relu',kernel_initializer=self.kernel_init,bias_initializer=self.bias_init),
                                        # layers.BatchNormalization(),
                                        layers.Dropout(drop_rate),
                                        layers.Dense(256, activation='tanh',kernel_initializer=self.kernel_init,bias_initializer=self.bias_init),
                                        # layers.Dropout(drop_rate),
                                        layers.Dense(output_size, activation='softmax')
                                        ])
        self.reg_net.summary()
        return self.reg_net

    @tf.function
    def train_step(self,x_batch,y_batch,X_test,Y_test):
        with tf.GradientTape() as tape:
            # forward propogtation
            y_pred = self.reg_net(x_batch)
            # calculate loss
            y_batch = tf.cast(tf.convert_to_tensor(y_batch), tf.float32)
            loss = self.cce(y_batch, y_pred)

        # calculate gradients with respect to trainable parameters
        gradients = tape.gradient(loss,self.reg_net.trainable_variables)

        # update network weights
        self.optimizer.apply_gradients(zip(gradients, self.reg_net.trainable_variables))

        # evaluation
        y_pred_train = self.reg_net(x_batch)
        y_pred_test = self.reg_net(X_test)

        self.m1.reset_state()
        self.m2.reset_state()
        train_acc = self.m1.update_state(y_batch,y_pred_train)
        y_test = tf.cast(tf.convert_to_tensor(Y_test), tf.float32)
        test_average_acc = self.m2.update_state(y_test,y_pred_test)


        return loss, self.m1.result(), self.m2.result()
    
    def train_loop(self,X_train,Y_train,X_test,Y_test):
        Loss = np.zeros((self.EPOCH_SIZE,1))
        Train_error = np.zeros((self.EPOCH_SIZE,1))
        Test_error = np.zeros((self.EPOCH_SIZE,1))

        for epoch in range(self.EPOCH_SIZE):
            idx = np.random.randint(0, X_train.shape[0], self.BATCH_SIZE)
            x_batch = X_train[idx,:]
            y_batch = Y_train[idx]
            loss, train_error, test_error = self.train_step(x_batch,y_batch,X_test,Y_test)

            # coverge speed
            # output the epoch while loss strat to be less than 0.1 and remain for 10 epoch
            if self.converge_epoch == 0:
                if loss < 0.1:
                    self.count = self.count + 1
                if loss >= 0.1:
                    self.count = 0
                if self.count > self.tol_counts:
                    self.converge_epoch = epoch - self.tol_counts

            # Plot the progress (D_loss, accuracy, G_loss)
            if epoch % 1000 == 0:
                print ("%d [loss: %s, train_accuracy: %s, test_accuracy: %s]" % (epoch, loss.numpy(), train_error.numpy(), test_error.numpy()))

            # save data for plotting
            Loss[epoch] = loss
            Train_error[epoch] = train_error
            Test_error[epoch] = test_error
        return Loss, Train_error, Test_error, self.converge_epoch

    def create_adversarial_pattern(self,x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self.reg_net(x)
            loss = self.cce(y, y_pred)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, x)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    def adversarial_training(self,X_train,Y_train,X_test,Y_test):
        Loss = np.zeros((self.adv_EPOCH_SIZE,1))
        Train_error = np.zeros((self.adv_EPOCH_SIZE,1))
        Test_error = np.zeros((self.adv_EPOCH_SIZE,1))

        for epoch in range(self.adv_EPOCH_SIZE):
            idx = np.random.randint(0, X_train.shape[0], self.BATCH_SIZE)
            x_batch = X_train[idx,:]
            y_batch = Y_train[idx]
            
            x_batch = tf.cast(tf.convert_to_tensor(x_batch), tf.float32)
            # training step
            perturbations = self.create_adversarial_pattern(x_batch, y_batch)
            adv_x = x_batch + self.epsilon*perturbations

            '''
            ## visulization
            image_normalized = x_batch[0,:].numpy()
            image = (image_normalized + 1) * 255/2
            image_mat = image.reshape(16,16)

            plt.imshow(image_mat, interpolation='nearest')
            plt.show()

            ## visulization
            image_normalized = adv_x[0,:].numpy()
            image = (image_normalized + 1) * 255/2
            image_mat = image.reshape(16,16)

            plt.imshow(image_mat, interpolation='nearest')
            plt.show()
            '''

            loss, train_error, test_error = self.train_step(adv_x,y_batch,X_test,Y_test)

            # Plot the progress (D_loss, accuracy, G_loss)
            if epoch % 1000 == 0:
                print ("%d [loss: %s, train_accuracy: %s, test_accuracy: %s]" % (epoch, loss.numpy(), train_error.numpy(), test_error.numpy()))

            # save data for plotting
            Loss[epoch] = loss
            Train_error[epoch] = train_error
            Test_error[epoch] = test_error
        return Loss, Train_error, Test_error


'''
if __name__ == '__main__':
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

    epoch_size = 50000
    batch_size = 3000

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
    Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)

    # network testing
    ############### 1. fully connected network ##############
    # define network
    net1 = My_network(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
    reg_net = net1.define_network(input_size = 256, output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_acc, Test_acc, converge_epoch = net1.train_loop(X_train_full,Y_train,X_test_full,Y_test)

    epoch_list = np.linspace(1,epoch_size, epoch_size)
    np.savetxt('train_acc.txt',Train_acc)
    np.savetxt('test_acc.txt',Test_acc)
    plt.plot(epoch_list,Train_acc,label='training accuracy')
    plt.plot(epoch_list,Test_acc,label ='test acuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
'''








