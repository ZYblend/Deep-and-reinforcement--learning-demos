import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi


import Prob3   # import functions in 'Prob3.py'

################## load trained network #############
conv_net = keras.models.load_model('my_model_conv.h5')

################## load dataset ################
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


################## load trained network #############
digit6 = Prob3.find_digit_from_dataset(6,conv_net,x_train,y_train_raw)
plt.imshow(digit6,cmap='gray')
plt.show()
vect_shape = digit6.shape[:2]


# 
patch = np.zeros([8,8,1])
patch_length = 8
patch_width  = 8
row_num = vect_shape[0] - patch_width + 1 
col_num = vect_shape[1] - patch_length + 1

table = np.linspace(1,8,8)
map1 = pd.DataFrame(columns = table)
map1.insert(0,'prob_of_6',table) 
map1 = map1.set_index('prob_of_6')
map2 = pd.DataFrame(columns = table)
map2.insert(0,'highest_prob',table) 
map2 = map2.set_index('highest_prob')
map3 = pd.DataFrame(columns = table)
map3.insert(0,'label',table) 
map3 = map3.set_index('label')

adv_num = 1
for iter1 in range(row_num):
    for iter2 in range(col_num):
        digit6_copy = digit6.copy()
        digit6_copy[iter1: (patch_width + iter1), iter2 :(patch_length + iter2), :] = patch

        input = np.expand_dims(digit6_copy, 0)
        ypred = conv_net(input)

        output = ypred.numpy()
        map1.loc[iter1, iter2] = output[0,6]                       # probability of '6'
        map2.loc[iter1, iter2] = output.max()                     # highest probability
        map3.loc[iter1, iter2] = tf.math.argmax(ypred,1).numpy()  # classified label

        '''
        # save all adversarial image
        if map3.loc[iter1, iter2] != 6:
            plt.imshow(digit6_copy, cmap='gray')
            plt.savefig('adv'+ str(adv_num)+'.png')
            plt.close()
            adv_num += 1
        '''
        

print('probability of 6', map1)
print('highest probability', map2)
print('classified label', map3)
dfi.export(map1, 'map1.png')
dfi.export(map2, 'map2.png')
dfi.export(map3, 'map3.png')


############# test the adversarial image ####################
digit6_copy = digit6.copy()
digit6_copy[11: 19, 5 :13, :] = patch
input = np.expand_dims(digit6_copy, 0)
ypred = conv_net(input)
print(tf.math.argmax(ypred,1).numpy())
plt.imshow(digit6_copy, cmap='gray')
plt.show()




        


    