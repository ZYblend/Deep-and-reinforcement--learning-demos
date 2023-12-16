import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers.core import Reshape
import os

"""
## Prepare the data
"""

class My_network(object):
    def __init__(self,optimizer,EPOCH_SIZE=300, BATCH_SIZE=32):
        # hyper-parameters
        self.EPOCH_SIZE = EPOCH_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self.optimizer = optimizer

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.m1 = tf.keras.metrics.CategoricalAccuracy()
        self.m2 = tf.keras.metrics.CategoricalAccuracy()

    def define_network(self,input_shape,output_shape):
        self.reg_net = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(12, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(12, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(output_shape, activation="softmax"),
            ])
        self.reg_net.summary()
        return self.reg_net
    
    @tf.function
    def train_step(self,x_batch,y_batch,X_test,Y_test):
        with tf.GradientTape() as tape:
            # forward propogtation
            y_pred = self.reg_net(x_batch)
            # calculate loss
            # y_batch = tf.cast(tf.convert_to_tensor(y_batch), tf.float32)
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
        self.m1.update_state(y_batch,y_pred_train)
        y_test = tf.cast(tf.convert_to_tensor(Y_test), tf.float32)
        self.m2.update_state(y_test,y_pred_test)

        return loss, self.m1.result(), self.m2.result()
    
    def train_loop(self,X_train,Y_train,X_test,Y_test):
        Loss = np.zeros((self.EPOCH_SIZE,1))
        Train_acc = np.zeros((self.EPOCH_SIZE,1))
        Test_acc = np.zeros((self.EPOCH_SIZE,1))

        for epoch in range(self.EPOCH_SIZE):
            idx = np.random.randint(0, X_train.shape[0], self.BATCH_SIZE)
            x_batch = X_train[idx,:]
            y_batch = Y_train[idx]
            loss, train_acc, test_acc = self.train_step(x_batch,y_batch,X_test,Y_test)

            # Plot the progress (D_loss, accuracy, G_loss)
            if epoch % 100 == 0:
                print ("%d [loss: %s, train_accuracy: %s, test_accuracy: %s]" % (epoch, loss.numpy(), train_acc.numpy(), test_acc.numpy()))

            # save data for plotting
            Loss[epoch] = loss
            Train_acc[epoch] = train_acc
            Test_acc[epoch] = test_acc

        return Loss, Train_acc, Test_acc

##################### pre-train a convolution network with accuracy bigger than 95% ####################################
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train_raw.astype("float32") / 255
    x_test = x_test_raw.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train_raw, num_classes)
    y_test = keras.utils.to_categorical(y_test_raw, num_classes)

    batch_size = 128
    epochs = 15000

    # training
    my_optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
    conv_class = My_network(optimizer=my_optimizer,EPOCH_SIZE=epochs, BATCH_SIZE=batch_size)

    conv_net = conv_class.define_network(input_shape, num_classes)

    # train network
    Loss, Train_error, Test_error = conv_class.train_loop(x_train,y_train,x_test,y_test)

    # save network
    conv_net.save('my_model_conv.h5')
    

    # plotting
    fig,ax = plt.subplots(1,3,figsize=(12,4))
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
    plt.show()


    # evaluation
    m1 = tf.keras.metrics.CategoricalAccuracy()
    for iter in range(6):
        Y_train_pred = conv_net(x_train[10000*(iter-1):iter*10000,:,:,:])
        m1.update_state(y_train[10000*(iter-1):iter*10000],Y_train_pred)
    print('Train set accuracy:',m1.result().numpy())

    Y_test_pred = conv_net(x_test)
    m2 = tf.keras.metrics.CategoricalAccuracy()
    m2.update_state(y_test,Y_test_pred)
    print('Test set accuracy:', m2.result().numpy())

 




   


    






    



