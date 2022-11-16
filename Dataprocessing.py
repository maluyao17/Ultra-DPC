import csv
import numpy as np
from sklearn import preprocessing
import scipy.io as sio
from sklearn.metrics.pairwise import pairwise_distances
import random
import h5py

def load_Data(filename):
    """
        load_Data: load '.csv' files
        returns: data of array inputs& original label of array inputs
    """
    data = []
    label = []
    with open(filename, 'r') as file_obj:
        file = csv.reader(file_obj)
        for row in file:
            per_data = []
            for i in row[:-1]:
                per_data.append(float(i))
            data.append(per_data)
            label.append(float(row[-1]))
    data_arr = np.array(data)
    label_arr = np.array(label, np.int32)
    # data_arr = normalization(data_arr)
    # mapminmax = preprocessing.MinMaxScaler()
    # data_arr = mapminmax.fit_transform(data_arr)
    return data_arr, label_arr

def load_Data2(filename):
    """
        load_Data2: load '.csv' files
        returns: data of array inputs
    """
    data = []
    with open(filename, 'r') as file_obj:
        file = csv.reader(file_obj)
        for row in file:
            per_data = []
            for i in row[:]:
                per_data.append(float(i))
            data.append(per_data)
    data_arr = np.array(data)
    # data_arr = normalization(data_arr)
    # mapminmax = preprocessing.MinMaxScaler()
    # data_arr = mapminmax.fit_transform(data_arr)
    return data_arr

def get_data(filename):
    """
        from yanggeping
        get_data: load '.mat' files
        returns:data of array inputs& original label of array inputs
    """
    # l= h5py.File(filename,'r','-v6')
    l = sio.loadmat(filename)
    data = l['fea']
    # print(type(data))
    # data = data.todense()
    # data = data.todense()
    # data = np.array(data)
    # print(data)
    label = l['gnd']
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return data ,np.array(label, np.int32)

def get_data2(filename):
    """
        from yanggeping
        get_data: load '.mat' files
        returns:data of array inputs& original label of array inputs
    """
    l= h5py.File(filename,'r')
    # l = sio.loadmat(filename)
    data = l['fea']
    # print(type(data))
    # data = data.todense()
    # data = data.todense()
    data = np.array(np.transpose(data))
    # print(data)
    label = l['gnd']
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return data ,np.array(label, np.int32)

def get_emnist(filename):
    l = sio.loadmat(filename)
    dataset = l['dataset']
    train =dataset['train'][0, 0]
    test = dataset['test'][0, 0]

    train_data = train['images'][0, 0]
    train_label = train['labels'][0, 0]

    test_data = test['images'][0, 0]
    test_label = test['labels'][0, 0]

    data = np.vstack([train_data, test_data])
    label = np.vstack([train_label, test_label])
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return np.ascontiguousarray(data) ,np.array(label, np.int32)


def Euc_dist(mat_a, mat_b):
    """
        Euc_dist: Calculate the Euclidean distance between mat_a and mat_b
    """
    distance_mat = pairwise_distances(mat_a, mat_b)
    # distance_mat = np.power(distance_mat,2)
    return distance_mat

def mapminmax(arr, ymax, ymin):
    """
        Normalization of data
        It comes from function 'Mapminmax()' in MATLAB
        params:
            arr: array inputs
            ymax: Upper bounds of the outputs
            ymin: lower bounds of the outputs
        returns:
            arr_return: array outputs
    """
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = np.reshape(arr, (arr.shape[0],1))
        arr = np.transpose(arr)
    arr_b = np.transpose(np.min(arr, axis=1))
    arr_b = np.reshape(arr_b, (arr.shape[0],1))
    arr_c = np.repeat(arr_b, arr.shape[1],axis=1)
    arr_d = arr - arr_c

    arr_e = np.transpose(np.max(arr, axis=1))
    arr_e = np.reshape(arr_e, (arr.shape[0], 1))
    arr_f = np.repeat(arr_e, arr.shape[1], axis=1)
    arr_g = arr_f - arr_c

    arr_return = np.multiply((ymax-ymin), arr_d)/(arr_g)+ymin
    find_nan = np.argwhere(np.isnan(arr_return))
    # find_nan = np.empty([1,2])
    # for i in range(0, 100):
    #     # print(arr_return.shape[0])
    #     find_nan_tmp = np.argwhere(np.isnan(arr_return[(int)(arr_return.shape[0]*i/100):(int)(arr_return.shape[0]*(i+1)/100),:]))
    #     # print((int)(arr_return.shape[0]*i/100))
    #     find_nan_tmp[:, 0] = (int)(arr_return.shape[0]*(i)/100) + find_nan_tmp[:, 0]
    #     find_nan = np.r_[find_nan, find_nan_tmp]
    # find_nan = find_nan[2:,:]

    for i in range(0, len(find_nan)):
        index = find_nan[i]
        arr_return[index[0], index[1]] = arr[index[0], index[1]]
    # print(arr_return)
    return arr_return

def normalize_by_minmax(data_sequence):
    """
        Normalization of data
        From yanggeping
    """
    min_v = np.min(data_sequence)
    max_v = np.max(data_sequence)
    range_v = max_v - min_v

    data_sequence = (data_sequence - min_v)/range_v

    return data_sequence
