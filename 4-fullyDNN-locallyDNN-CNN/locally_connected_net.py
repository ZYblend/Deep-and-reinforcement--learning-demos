import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import pandas as pd

class My_network(object):
    def __init__(self,optimizer,EPOCH_SIZE=300, BATCH_SIZE=36, kernel_init = 'random_normal', bias_init = 'zeros'):
        # hyper-parameters
        self.EPOCH_SIZE = EPOCH_SIZE
        self.BATCH_SIZE = BATCH_SIZE

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
    
    def accuracy_metric(self,x,y):
        y_pred = self.reg_net(x).numpy()

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
                                        # layers.LocallyConnected2D(16, (3,3),activation='relu',kernel_initializer='normal', input_shape=input_size),
                                        # layers.LocallyConnected2D(12, (3,3),activation='relu',kernel_initializer='normal'),
                                        # layers.LocallyConnected2D(6, (3,3),activation='tanh',kernel_initializer='normal'),
                                        layers.LocallyConnected2D(8, (3,3),activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, input_shape=input_size),
                                        layers.LocallyConnected2D(6, (3,3),activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init),
                                        layers.LocallyConnected2D(6, (3,3),activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init),
                                        # layers.LocallyConnected2D(4, (3,3),activation='tanh',kernel_initializer='normal'),
                                        # layers.LocallyConnected2D(6, (2,2),activation='relu',kernel_initializer='normal'),
                                        # layers.LocallyConnected2D(4, (2,2),activation='tanh',kernel_initializer='normal'),
                                        layers.Flatten(),
                                        layers.Dense(300, activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init),
                                        layers.Dense(100, activation='tanh',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init),
                                        layers.Dense(output_size, activation='softmax')
                                        ])
        self.reg_net.summary()
        return self.reg_net
        # 1(8,3,3)2(8,3,3)3(6,3,3)4(300)5(100,tanh) 68%
        # 1(8,3,3)2(8,3,3)3(6,3,3)4(4,3,3,tanh) 32%
        # 1(8,3,3)2(6,3,3)3(6,3,3)4(4,3,3,tanh) 
        # 1(8,3,3)2(8,3,3)3(6,3,3,tanh) 59%
        # 1(8,4,4)2(8,3,3)3(6,3,3,tanh) 65%
        # 1(8,2,2)2(8,2,2)3(6,2,2,tanh) NO
        # 1(16,3,3)2(8,3,3)3(6,3,3,tanh) NO
        # 1(16,2,2)2(8,2,2)3(6,2,2,tanh) NO

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
                if loss < 0.5:
                    self.count = self.count + 1
                if loss >= 0.5:
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

'''
if __name__ == '__main__':
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
    ## visulization
    # id = train_data[0,0]
    # image_normalized = train_data[0,1:257]
    # image = (image_normalized + 1) * 255/2
    # image_mat = image.reshape(16,16)

    # plt.imshow(image_mat, interpolation='nearest')
    # plt.title('a sample')
    # plt.show()

    # hyper-parameters
    CLASSES_SIZE = 10   # number of classes for MNIST
    LR = 0.1   # learning rate
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR)

    epoch_size = 10000
    batch_size = 32

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, CLASSES_SIZE)
    Y_test = np_utils.to_categorical(Y_test, CLASSES_SIZE)

    ##########################################  Task 1 ###############################################
    # # define network
    net = My_network(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
    reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

    # Train network
    Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train,Y_train,X_test,Y_test)

    # save network
    reg_net.save('my_model_local_connectted.h5')

    # evaluation (accuracy)
    Y_train_pred = reg_net(X_train)
    m1 = tf.keras.metrics.CategoricalAccuracy()
    train_acc = m1.update_state(Y_train,Y_train_pred)
    print('Train set accuracy:',m1.result().numpy())

    Y_test_pred = reg_net(X_test)
    m2 = tf.keras.metrics.CategoricalAccuracy()
    test_acc = m2.update_state(Y_test,Y_test_pred)
    print('Test set accuracy:', m2.result().numpy())

    # evaluation (accuracy custom)
    train_acc2 = net.accuracy_metric(X_train, Y_train)
    test_acc2  = net.accuracy_metric(X_test, Y_test)
    print('Train set accuracy2:',train_acc2)
    print('Test set accuracy2:',test_acc2)


    # plotting
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
    ax[0].plot(Loss)
    ax[0].set_title('Loss curve')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(Train_error)
    ax[1].set_title('Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[2].plot(Test_error)
    ax[2].set_title('Testing Accuracy')
    ax[2].set_xlabel('Epochs')
    plt.savefig("Curve_locally_connected_"+str(epoch_size)+".png")
    
    plt.show()


    ####################################### Task 2 ########################################################
    ############# 1. initializer ################
    inititalizers_list = ['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'variance_scaling']
    tot1 = len(inititalizers_list)

    # create a void table
    Converge_epoch = pd.DataFrame(columns = inititalizers_list)
    Converge_epoch.insert(0,'initializer_name',inititalizers_list) 
    Converge_epoch = Converge_epoch.set_index('initializer_name')

    # test
    for idx1 in range(tot1):
        for idx2 in range(tot1):
            net = My_network(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size,kernel_init = inititalizers_list[idx1], bias_init = inititalizers_list[idx2])
            reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)
            Loss, Train_error, Test_error, Converge_epoch.loc[inititalizers_list[idx1],inititalizers_list[idx2]] = net.train_loop(X_train,Y_train,X_test,Y_test)
    
    #
    Converge_epoch.to_csv('Task2_Initilizer_locally_connected.csv')
    print('testing learning speed by using different initializer', Converge_epoch)

    
    ## 2. Learning rate
    LR_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]   # learning rate
    # LR_list = [ 0.2, 0.6, 1.0, 1.4, 1.8, 2]
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
    labels = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
    # labels = [ '0.2', '0.6', '1.0', '1.4','1.8','2']
    
    for iter in range(len(LR_list)):
        my_optimizer=tf.keras.optimizers.SGD(learning_rate=LR_list[iter])
        # define network
        net = My_network(optimizer=my_optimizer,EPOCH_SIZE = epoch_size, BATCH_SIZE = batch_size)
        reg_net = net.define_network(input_size = (16,16,1), output_size = CLASSES_SIZE)

        # Train network
        Loss, Train_error, Test_error, converge_epoch = net.train_loop(X_train,Y_train,X_test,Y_test)
        
        ax[0].plot(Loss,label=labels[iter])
        ax[0].set_title('Loss curve')
        ax[0].set_xlabel('Epochs')
        ax[0].legend()
        ax[1].plot(Train_error,label=labels[iter])
        ax[1].set_title('Training Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].legend()
        ax[2].plot(Test_error,label=labels[iter])
        ax[2].set_title('Testing Accuracy')
        ax[2].set_xlabel('Epochs')
        ax[2].legend()

    plt.savefig("learning_rate2_local.png")
    plt.show()
    '''

    ## 






