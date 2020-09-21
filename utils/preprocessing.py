import math
import os
import numpy as np
import pandas

def count_rates(test_e):
    ids=np.unique(test_e) #array of unique ids
    event_rate=np.array([len(test_e[test_e==i]) for i in ids])/len(test_e)
    return(event_rate)

def loadDataDict(name, file_path):
    x = np.load(file_path+'{}_x'.format(name)+'.npy')
    e = np.load(file_path+'{}_e'.format(name)+'.npy')
    t = np.load(file_path+'{}_t'.format(name)+'.npy')
    pat = np.load(file_path+'{}_pat'.format(name)+'.npy')
    return({'x':x, 'e':e, 't':t, 'pat':pat})

# delete censored objects before time cut
def datadicTimeCut_delcensor(data_dic, time_cut=168):
    
    keep_idx = np.where((data_dic['e']==1 ) | (data_dic['t'] > time_cut))
    print("keeped proportion:{}".format(len(keep_idx[0])/len(data_dic['e'])))
    print(" x_shape:{}".format(data_dic['x'][keep_idx].shape))
    new_dic = {'x':data_dic['x'][keep_idx], 'e':data_dic['e'][keep_idx], 't':data_dic['t'][keep_idx]}
    
    # anyone with t>time_cut has label 0
    censor_idx = np.where(new_dic['t'] > time_cut)
    # set the corresponding event label as 0
    new_dic['e'][censor_idx] = 0
    # set the corresponding time as time_cut
    new_dic['t'][censor_idx] = time_cut    
    end_time = max(data_dic['t'][keep_idx])
    print(" end_time:{}".format(end_time))
    print(" observed percent:{}".format(sum(new_dic['e']) / len(new_dic['e'])))

    return(new_dic)


def datadicTimeCut_rightcensor(data_dic, time_cut=168, time_cut_low=0):
    keep_idx = np.where((data_dic['t'] > time_cut_low)&(data_dic['t'] <= time_cut))
    print(" x_shape:{}".format(data_dic['x'][keep_idx].shape))
    end_time = max(data_dic['t'][keep_idx])
    print(" end_time:{}".format(end_time))
    print(" observed percent:{}".format(sum(data_dic['e'][keep_idx]) / len(data_dic['e'][keep_idx])))

    return({'x':data_dic['x'][keep_idx], 'e':data_dic['e'][keep_idx], 't':data_dic['t'][keep_idx]})

# do not delete objects! just cut observations at certain points and count those as censor
def datadicTimeCut(data_dic, time_cut=168):
    print(" number of observations:{}".format(data_dic['x'].shape[0]))
    end_time = max(data_dic['t'])
    # select indices which have event/censoring time after time_cut
    censor_idx = np.where(data_dic['t'] > time_cut)
    print(" number of observations have event/censoring time after time {}:{}".format(time_cut, data_dic['x'][censor_idx].shape[0]))
    # set the corresponding event label as 0
    data_dic['e'][censor_idx] = 0
    # set the corresponding time as time_cut
    data_dic['t'][censor_idx] = time_cut
    
    
    event_idx = np.where(data_dic['e'] == 1)
    print(" end_time:{}".format(end_time))
    print(" largest observed time :{}".format(max(data_dic['t'][event_idx])))
    print(" observed percent:{}".format(sum(data_dic['e']) / len(data_dic['e'])))

    return({'x':data_dic['x'], 'e':data_dic['e'], 't':data_dic['t']})

def flatten_nested(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened

# one-hot-encoding all categorical variables
def one_hot_encoder(data, encode):
    data_encoded = data.copy()
    encoded = pandas.get_dummies(data_encoded, prefix=encode, columns=encode)
#    print("head of data:{}, data shape:{}".format(data_encoded.head(), data_encoded.shape))
#    print("Encoded:{}, one_hot:{}{}".format(encode, encoded.shape, encoded[0:5]))
    return encoded


# return column indices for one columns
def one_hot_indices(dataset, one_hot_encoder_list):
    indices_by_category = []
    for column in one_hot_encoder_list:
        values = dataset.filter(regex="{}_.*".format(column)).columns.values
        # print("values:{}".format(values, len(values)))
        indices_one_hot = []
        for value in values:
            indice = dataset.columns.get_loc(value)
            # print("column:{}, indice:{}".format(colunm, indice))
            indices_one_hot.append(indice)
        indices_by_category.append(indices_one_hot)
    # print("one_hot_indices:{}".format(indices_by_category))
    return indices_by_category

# for training/testing/validation split
def formatted_data_original(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data

def formatted_data_ash(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data
# for training/testing/validation split - non-survival
def formatted_data(x, e, idx):
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    dic_data = {'x': covariates, 'e': censoring}
    return dic_data

def missing_proportion(dataset):
    missing = 0
    columns = np.array(dataset.columns.values)
    for column in columns:
        missing += dataset[column].isnull().sum()
    return 100 * (missing / (dataset.shape[0] * dataset.shape[1]))

def get_train_median_mode(x, categorial):
    categorical_flat = flatten_nested(categorial)
    print("categorical_flat:{}".format(categorical_flat))
    imputation_values = []
    print("len covariates:{}, categorical:{}".format(x.shape[1], len(categorical_flat)))
    median = np.nanmedian(x, axis=0)
    mode = []
    for idx in np.arange(x.shape[1]):
        a = x[:, idx]
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_idx = a[index]
        mode.append(mode_idx)
    for i in np.arange(x.shape[1]):
        if i in categorical_flat:
            imputation_values.append(mode[i])
        else:
            imputation_values.append(median[i])
    print("imputation_values:{}".format(imputation_values))
    return imputation_values

def saveDataCSV(data_dic, name, file_path):
    df_x = pandas.DataFrame(data_dic['x'])
#     one_hot_indices = data_dic['one_hot_indices']
#     # removing 1 column for each dummy variable to avoid colinearity
#     if one_hot_indices:
#         rm_cov = one_hot_indices[:,-1]
#         df_x = df_x.drop(columns=rm_cov)
    df_e = pandas.DataFrame({'e':data_dic['e']})
    df = pandas.concat([df_e, df_x], axis=1, sort=False)
    df.to_csv(file_path+'/'+name+'.csv', index=False)
    
def loadDataCSV(name, file_path):
    df = pandas.read_csv(file_path+'/'+name+'.csv')
    n_total = df.shape[1]
    z_dim = 4
    df_x = df.iloc[:,range(1,n_total)]
    df_e = df.iloc[:,0]
    return({'x':np.array(df_x), 'e':np.array(df_e)})
    

