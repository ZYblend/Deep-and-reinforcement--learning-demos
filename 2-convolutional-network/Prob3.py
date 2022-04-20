import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Model

def find_digit_from_dataset(digit_label,model,x,y_raw):
    ''' Find the required digit number from dataset
         and make sure it is clssified correctly by the trained network'''
    idx = 0
    ypred = np.array([digit_label+1])
    while ypred[0] != digit_label:
        while y_raw[idx] != digit_label:
            idx += 1
            digit = x[idx,:,:,:]
        input = np.expand_dims(digit, 0)
        y_pred = model(input)
        ypred = tf.math.argmax(y_pred,1).numpy()
    # print('This digit', digit_label, 'is successfully recognized!')
    return digit

def visualization_filter(img_ix,model):
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
        # nomalize filters
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot
        ix = 1
        plt.figure(img_ix)
        for i in range(filters.shape[3]):
            f = filters[:, :, :, i]
            for j in range(filters.shape[2]):
                ax = plt.subplot(filters.shape[3], filters.shape[2], ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f[:, :, j], cmap='gray')

                ix += 1
        img_ix += 1
    return img_ix

def Get_feature_maps(model,input,img_INIix):
    input = np.expand_dims(input, 0)
    feature_maps = []
    img_ix = 1
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        tmp_model = Model(model.layers[0].input, layer.output)
        tmp_output = tmp_model.predict(input)[0]
        feature_maps.append(tmp_output)
        ix = 1
        plt.figure(img_INIix + img_ix)
        for i in range(int(tmp_output.shape[2]/4)):
            for j in range(4):
                ax = plt.subplot(int(tmp_output.shape[2]/4), 4, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(tmp_output[:, :, ix-1], cmap='gray')
                ix += 1
        img_ix += 1
    return img_ix

def Pad_vector(vector, how, depth, constant_value=0):
    vect_shape = vector.shape[:2]
    if (how == 'left'):
        pp = np.full(shape=(vect_shape[0], depth), fill_value=vector[:,vect_shape[1]-depth:vect_shape[1]])
        pv = np.hstack(tup=(vector[:,depth:vect_shape[1]], pp))
    elif (how == 'right'):
        pp = np.full(shape=(vect_shape[0], depth), fill_value=vector[:,0:depth])
        pv = np.hstack(tup=(pp, vector[:,0:vect_shape[1]-depth]))
    else:
        return vector
    return pv

if __name__ == '__main__':
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
    

    ################## (1) Visualization of filters #######################
    print('shape of filters:')
    img_endix = visualization_filter(0,conv_net)

    ################## (2) visualization of feature map ####################
    # find test cases: digit '0' and digit '8'
    digit0 = find_digit_from_dataset(0,conv_net,x_test,y_test_raw)
    digit8 = find_digit_from_dataset(8,conv_net,x_test,y_test_raw)

    # plot feature maps
    img_endix = Get_feature_maps(conv_net,digit0,2)
    img_endix = Get_feature_maps(conv_net,digit8,5)


    ################## (3) visualization of feature map ####################
    digit7 = find_digit_from_dataset(7,conv_net,x_train,y_train_raw)
    plt.figure(10)
    plt.imshow(digit7, cmap='gray')
    plt.title('before shifting')

    # shift with clamp border padding method
    temp_digit7 = Pad_vector(digit7[:,:,0], 'left', 5)
    new_digit7  = Pad_vector(temp_digit7, 'right', 5)
    new_digit7 = np.expand_dims(new_digit7,2)
    plt.figure(11)
    plt.title('after shifting')
    plt.imshow(new_digit7, cmap='gray')
   
    
    # test with conv network
    input = np.expand_dims(new_digit7, 0)
    ypred = conv_net(input)
    label_pred = tf.math.argmax(ypred,1).numpy()
    if label_pred == 7:
        print('successfully classified after shifting!')
    else:
        print('wrongly classified after shifting!')

    plt.show()

    

