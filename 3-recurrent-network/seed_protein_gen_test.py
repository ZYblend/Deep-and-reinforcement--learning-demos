from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sample_output(sfm_out):
    sfm_out = sfm_out.reshape([99,21])
    indice_out = np.argmax(sfm_out,axis=1)
    str_out = ''
    for i in range(num_samples-1): str_out += character_list[indice_out[i]]
    return indice_out, str_out

def find_max_match(indice_p, indice_r, fixed_num):
    return_idc = 0
    for max_match in reversed(range(20)):
        for idx in range(fixed_num-1, num_samples-1-max_match):
            if indice_r[idx+max_match-1] == 20:
                break
            if np.array_equiv (indice_r[idx:(idx+max_match)], indice_p[idx:(idx+max_match)]):
                return_idc = 1
                if max_match == 19:
                    print(indice_r[idx:(idx+max_match)])
                break
        if return_idc == 1:
            break
    return idx, max_match    
        

total_num = 4000  # 300
test_num = int(total_num/5)
trunc_size = 100
character_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','0']
charlist_size = len(character_list)
with open('pdb_seqres.txt') as f:
    counter = 1
    test_sqeuence_list = []
    while counter<=total_num:
        array_line=np.zeros((trunc_size, charlist_size))
        line = f.readline()
        if counter%5 == 0: 
            in_line = line[0:trunc_size].ljust(trunc_size,'0')
            in_line = in_line.replace("\n","").ljust(trunc_size,'0')
            for i in range(len(in_line)): array_line[i,character_list.index(in_line[i])]=1
            test_sqeuence_list.append(array_line)
        counter += 1
test_array = np.zeros((test_num,trunc_size,charlist_size))
for i in range(test_num):
    array_tmp = test_sqeuence_list[i]
    test_array[i,:,:] = array_tmp.reshape(1,trunc_size, charlist_size)

validation_data_list = test_array
num_vali = validation_data_list.shape[0]
print(num_vali, "test sequences")

X_test  = validation_data_list[:,0:trunc_size-1,:]
Y_test  = validation_data_list[:,1:trunc_size,:]

num_samples  = trunc_size
num_features = charlist_size

my_model =  keras.models.load_model('my_model_lstm.h5')
sequence_gen_table = np.zeros((10,20))
for k in range(1, 11):
    Gen_data_list = np.zeros((test_num,trunc_size,charlist_size))
    Gen_data_list[:,0:k,:] = validation_data_list[:,0:k,:]
    Gen_data_list[:,k:,-1] = 1
    Gen_test = Gen_data_list[:,0:trunc_size-1,:]
    for order in range(test_num):
        in_Gen_test = Gen_test[order,:,:]
        in_Gen_indice = np.argmax(in_Gen_test.reshape([num_samples-1, num_features]),axis=1)
        # print(in_Gen_indice)
        pred_res_prob = my_model.predict(in_Gen_test.reshape([1,99,21]))
        pred_res_indice, pred_res_str = sample_output(pred_res_prob)
        # print(pred_res_indice)
        true_indice = np.argmax(Y_test[order,:,:].reshape([num_samples-1, num_features]),axis=1)
        # print(true_indice)
        loc_start, matched_len = find_max_match(pred_res_indice, true_indice, k)
        # print("order=",order, "matched length=", matched_len)
        if matched_len>=19: matched_len = 19
        sequence_gen_table[k-1,matched_len] += 1
    print(k)
print(sequence_gen_table)
np.savetxt('sqeuence_gen_table.txt',sequence_gen_table,fmt='%d')

# my_model.evaluate(Gen_test, Y_test)