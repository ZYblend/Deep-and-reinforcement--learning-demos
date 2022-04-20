from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

def sample_output(sfm_out):
    # sfm_out = sfm_out.reshape([99,21])
    indice_out = np.argmax(sfm_out,axis=1)
    str_out = ''
    for i in range(sfm_out.shape[0]): str_out += character_list[indice_out[i]]
    return indice_out, str_out

############################ load dataset ##############################################
total_num = 4000  # 300
train_num = int(total_num*4/5)
test_num = int(total_num/5)
trunc_size = 100
character_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','0']
charlist_size = len(character_list)
with open('pdb_seqres.txt') as f:
    counter = 1
    train_sqeuence_list = []
    test_sqeuence_list = []
    while counter<=total_num:
        array_line=np.zeros((trunc_size, charlist_size))
        line = f.readline()
        in_line = line[0:trunc_size].ljust(trunc_size,'0')
        in_line = in_line.replace("\n","").ljust(trunc_size,'0')
        for i in range(len(in_line)): array_line[i,character_list.index(in_line[i])]=1
        if counter%5 == 0: test_sqeuence_list.append(array_line)
        else: train_sqeuence_list.append(array_line)
        counter += 1
train_array = np.zeros((train_num,trunc_size,charlist_size))
test_array = np.zeros((test_num,trunc_size,charlist_size))
for i in range(train_num):
    array_tmp = train_sqeuence_list[i]
    train_array[i,:,:] = array_tmp.reshape(1,trunc_size, charlist_size)
for i in range(test_num):
    array_tmp = test_sqeuence_list[i]
    test_array[i,:,:] = array_tmp.reshape(1,trunc_size, charlist_size)


################################ prepare datset ##########################################
train_data_list = train_array
validation_data_list = test_array
num_train = train_data_list.shape[0]
num_vali = validation_data_list.shape[0]
print(num_train, "train sequences")
print(num_vali, "test sequences")

X_train = train_data_list[:,0:99,:]
Y_train = train_data_list[:,1:100,:]

X_test  = validation_data_list[:,0:99,:]
Y_test  = validation_data_list[:,1:100,:]


################################## model training #########################################
# hyperparameter
num_samples  = 100
num_features = 21
lstm_units = 512
batch_size = 32
epochs = 300

model = keras.Sequential(
    [
        keras.Input(shape=(num_samples-1, num_features)),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dense(num_features, activation="softmax"),
    ]
)
model.summary()
# lossfunc = tf.keras.metrics.CategoricalAccuracy()
model.compile(optimizer='RMSprop',loss="categorical_crossentropy",metrics=[tf.keras.metrics.CategoricalAccuracy()])

Loss = np.zeros((epochs,1))
acc = np.zeros((epochs,1))
val_loss = np.zeros((epochs,1))
val_acc = np.zeros((epochs,1))
for epoch in range(epochs):
    print("epoch:", epoch)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1,validation_data=(X_test, Y_test))

    Loss[epoch,:] = history.history['loss']
    acc[epoch,:] = history.history['categorical_accuracy']

    val_loss[epoch,:] = history.history['val_loss']
    val_acc[epoch,:] = history.history['val_categorical_accuracy']


model.save('my_model_lstm.h5')

# plotting
fig,ax = plt.subplots(2,2,figsize=(12,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15,wspace=0.3,hspace=0.15)
ax[0,0].plot(Loss)
ax[0,0].set_title('Training Loss')
ax[0,0].set_xlabel('Epochs')
ax[0,1].plot(val_loss)
ax[0,1].set_title('Validation Loss')
ax[0,1].set_xlabel('Epochs')
ax[1,0].plot(acc)
ax[1,0].set_title('Training Accuracy')
ax[1,0].set_xlabel('Epochs')
ax[1,1].plot(val_acc)
ax[1,1].set_title('Validation Accuracy')
ax[1,1].set_xlabel('Epochs')
plt.show()

res1 = model.predict(X_train[:,0:99,:])
m1 = tf.keras.metrics.CategoricalAccuracy()
m1.update_state(Y_train,res1)
print('Fianl train accuracy',m1.result())

res2 = model.predict(X_test[:,0:99,:])
m2 = tf.keras.metrics.CategoricalAccuracy()
m2.update_state(Y_test,res2)
print('Fina test accuracy',m2.result())


################################## Analyze long-term dependencies ###############################
# model = keras.models.load_model('my_model_lstm.h5')
test_sequence = train_data_list[1]
indice_sample, str_sample = sample_output(test_sequence)
print('Test sample',str_sample)
print('Test sample indice',indice_sample)
test_sequence = test_sequence.reshape(1,test_sequence.shape[0],test_sequence.shape[1])
test_input = test_sequence[:,0:99,:]
test_output = test_sequence[:,1:100,:]
#print('test output',test_output)
res = model.predict(test_input)
#print('test predicted output', res)

# randomly change one cahracter
n_iteration = 100  # pererform Monte Carlo experiment
num_char = 1
Diff_average = np.zeros(test_sequence.shape[1])
for iter in range(n_iteration):
    diff_average = np.zeros(test_sequence.shape[1])
    for idx in range(test_sequence.shape[1]):
        modified_idx = indice_sample.copy()
        # idx = np.random.randint(0, num_samples-1, num_char)
        modified_idx[idx] = np.random.randint(0, num_features-1, num_char)

        modified_seq = np_utils.to_categorical(modified_idx, num_features)
        modified_seq = modified_seq.reshape(1,modified_seq.shape[0],modified_seq.shape[1])

        modified_input = modified_seq[:,0:99,:]
        res_modified = model.predict(modified_input)
        diff = res_modified-res
        diff = np.squeeze(diff, axis=0)
        diff_average[idx] = np.linalg.norm(np.sum(diff,axis=0)/99)
    Diff_average +=diff_average
        
idx_withChange = np.argwhere(Diff_average/n_iteration >= 0.05)

print('The lonest dependencies',idx_withChange)


################################# 3-gram language models ###########################
## All possible 3-gram acids sequence 20*20*20=8000 possible sequences
num_seq = 20**3
one_seq = np.linspace(0,19,20)
All_3gram_seq = np.array(np.meshgrid(one_seq, one_seq, one_seq)).T.reshape(-1,3)
print(num_seq)
print(All_3gram_seq.shape)

## Network
def network_3gram_model(All_3gram_seq,num_gram = 3400):
    ''' num_gram * 97 sequence in total
    '''
    model = keras.models.load_model('my_model_lstm.h5')
    input = np.zeros((num_gram,99,21))
    for idx in range(input.shape[0]):
        indice = np.random.randint(0,21,3)
        seq = np_utils.to_categorical(indice, num_features)
        input[idx,:3,:] = seq
    all_seq = model.predict(input)

    all_3gram_3d = np.zeros((all_seq.shape[0],(all_seq.shape[1]-2),3))
    for idx in range(all_seq.shape[0]):
        indice_out, str_out = sample_output(all_seq[idx])
        for idx2 in range(all_3gram_3d.shape[1]):
            all_3gram_3d[idx,idx2,0] = indice_out[idx2]
            all_3gram_3d[idx,idx2,1] = indice_out[idx2+1]
            all_3gram_3d[idx,idx2,2] = indice_out[idx2+2]
    all_3gram_net = all_3gram_3d.reshape(all_3gram_3d.shape[0]*all_3gram_3d.shape[1],all_3gram_3d.shape[2])  # all 3 grams generated by network

    P = np.zeros(All_3gram_seq.shape[0])
    '''
    for iter1 in range(All_3gram_seq.shape[0]):
        for iter2 in range(all_3gram_net.shape[0]):
            if np.array_equal(All_3gram_seq[iter1,:], all_3gram_net[iter2,:]):
                P[iter1] += 1
    '''
    for iter in range(All_3gram_seq.shape[0]):
        All_3gram_seq_mat = np.repeat(np.expand_dims(All_3gram_seq[iter,:],-1).T,all_3gram_net.shape[0],axis=0)
        P[iter] = np.sum((1*(All_3gram_seq_mat == all_3gram_net)).all(axis=1))
    P = P/all_3gram_net.shape[0]
    # print(np.sum(P))
    np.savetxt('network_p.txt',P)
    return P

## training dataset
def train_3gram_model():
    train_array_indice = np.argmax(train_array, axis=2)
    count_gram = np.zeros((20,20,20))
    for idx in range(test_num):
        for i in range(trunc_size-3):
            gram_tmp = train_array_indice[idx,i:i+3]
            if gram_tmp[-1] == 20:
                break
            count_gram[gram_tmp[0],gram_tmp[1],gram_tmp[2]]+=1
    valid_gram_sum = sum(sum(sum(count_gram)))
    count_gram_axised = np.moveaxis(count_gram, -1, 0)
    p_count_gram = count_gram_axised.reshape((8000,1))/valid_gram_sum
    np.savetxt('dataset_p.txt',p_count_gram)
    return p_count_gram

## Compare and plotting
P_net = network_3gram_model(All_3gram_seq).reshape((8000,1))
P_train = train_3gram_model().reshape((8000,1))


P_diff = np.linalg.norm(P_net-P_train,axis=1)
plt.plot(P_diff)

max20_P_diff = np.argsort(P_diff)[-20:]
min20_P_diff = np.argpartition(P_diff, kth=1)[:20]

max20_seq = All_3gram_seq[max20_P_diff]
min20_seq = All_3gram_seq[min20_P_diff]

max20_list =[]
for i in range(max20_seq.shape[0]):
    str_out = ''
    for j in range(max20_seq.shape[1]):
        str_out += character_list[int(max20_seq[i,j])]
    max20_list.append(str_out)

min20_list =[]
for i in range(min20_seq.shape[0]):
    str_out = ''
    for j in range(min20_seq.shape[1]):
        str_out += character_list[int(min20_seq[i,j])]
    min20_list.append(str_out)

print('20 most different entries',max20_list)
print('20 most closely matched entries',min20_list)

plt.show()

plt.plot(P_net,'k-',label='all probability')
plt.plot(max20_P_diff,P_net[max20_P_diff],'ro', label ='max20')
plt.plot(min20_P_diff,P_net[min20_P_diff],'bo', label ='min20')
plt.ylim([0,0.025])
plt.legend()
plt.title('network probability')
plt.show()

plt.plot(P_train,'k-',label='all probability')
plt.plot(max20_P_diff,P_train[max20_P_diff],'ro', label ='max20')
plt.plot(min20_P_diff,P_train[min20_P_diff],'bo', label ='min20')
plt.ylim([0,0.025])
plt.legend()
plt.title('dataset probability')
plt.show()

plt.plot(P_net[max20_P_diff], label ='network')
plt.plot(P_train[max20_P_diff], label ='dataset')
plt.legend()
plt.title('proability corresponding to max difference')
plt.show()


plt.plot(P_net[min20_P_diff], label ='network')
plt.plot(P_train[min20_P_diff], label ='dataset')
plt.legend()
plt.title('proability corresponding to min difference')
plt.show()



