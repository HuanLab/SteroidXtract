#########################################
# This script is to apply SteroidXtract on LC-MS raw data files in mzXML format.
# Shipei Xing, Sep 22, 2020
# Copyright @ The University of British Columbia
#########################################
# Working directories
input_dir = 'E:/SteroidXtract_2020/20200320-steroid_liver&feces&gall-rp+/20200320-steroid-liver&feces&gall-rp+/std&IS/mzXML files_DataAnalysis' # input data path for SteroidXtract
output_dir = 'E:/SteroidXtract_2020/20200320-steroid_liver&feces&gall-rp+/20200320-steroid-liver&feces&gall-rp+/std&IS/mzXML prediction_binary steroid_20200803'
model_dir = 'E:/SteroidXtract_2020/Final codes & file templates'

# Parameter setting
ms1_tol = 0.005  # MS1 mass tolerance
ms2_tol = 0.01  # MS2 mass tolerance
rt_threshold = 23  # valid retention time in minute, MS2 after rt_threshold are discarded
pre_int_threshold = 1000  # precursor intensity threshold (MS2 with precursor intensity lower than the threshold are discarded)

#########################################
# Libraries
from pyteomics import mzxml, auxiliary, mzml, mgf
import numpy as np
import pandas as pd
import os
import math
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
print('Libraries loaded')

start_value = 50  # m/z range
end_value = 500
bin_width = 0.1  # bin width, for machine learning matrix generation
bin_No = (end_value - start_value) / bin_width

# delete elements(list) from dict
def removekey(d, key):
    r = dict(d)
    for i in range(len(key)):
        del r[key[i]]
    return r

# machine learning_functions
def dataset_passing_in(data, bw):
    dataset = pd.DataFrame(data)
    if bw == 0.1:
        # dataset.rename(columns = {dataset.columns[4500]: 'class'}, inplace = True)
        X = pd.DataFrame(data=dataset.iloc[:, 0:4500]).values
        # y = pd.DataFrame(data = dataset.loc[:, 'class']).astype('str').values
        return X


# load model
os.chdir(model_dir)
json_file = open('SteroidXtract_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("SteroidXtract_model.h5")
loaded_model.compile(optimizer='adam', loss='binary_crossentropy')

os.chdir(input_dir)
files = [f for f in os.listdir(input_dir) if f.endswith('.mzXML')]

for l in range(len(files)):
    print('New file loaded')
    os.chdir(input_dir)
    # read mzxml file
    mzxml_file = files[l]
    print(files[l])
    file = mzxml.MzXML(mzxml_file)  # dict
    feature_df = pd.DataFrame(np.nan, index=range(len(file)),
                              columns=['mzxml_index', 'precursor_MZ', 'rt', 'precursor_intensity'])

    # fill in precursorMZ and RT information
    h = 0
    for i in range(len(file)):
        if (file[i]['msLevel'] != 2): continue  # only MS2 recorded
        if (file[i]['retentionTime'] > rt_threshold): continue
        feature_df.iloc[h, 0] = int(file[i]['num'])
        feature_df.iloc[h, 1] = float(file[i]['precursorMz'][0]['precursorMz'])
        feature_df.iloc[h, 2] = float(file[i]['retentionTime'])
        feature_df.iloc[h, 3] = int(file[i]['precursorMz'][0]['precursorIntensity'])
        h = h + 1
    feature_df.dropna(subset=["mzxml_index"], inplace=True)

    # create data matrix for machine learning
    mass_df = pd.DataFrame(0, index=list(range(int(feature_df.shape[0]))), columns=list(range(int(bin_No + 1))))
    df = pd.concat([feature_df, mass_df], axis=1)

    # mass binning
    for i in range(df.shape[0]):
        index = df.iloc[i, 0].astype(int).item() - 1
        premass = df.iloc[i, 1].item()
        # premass > 233.22 (simplest steroid)
        if premass <= 233.22:
            continue
        # precursor intensity > pre_int_threshold:
        if df.iloc[i, 3].item() <= pre_int_threshold:
            continue
        # read MS2
        ms2 = np.concatenate((file[index]['m/z array'].reshape(file[index]['m/z array'].shape[0], 1),
                              file[index]['intensity array'].reshape(file[index]['intensity array'].shape[0], 1)),
                             axis=1)
        # remove fragments > premass
        ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] > (premass + ms1_tol)), axis=0)
        # at least 5 fragments
        if ms2.shape[0] < 5:
            continue
        # top 20 fragments
        if ms2.shape[0] > 20:
            int_threshold = -np.sort(-ms2[:, 1])[19]
            ms2 = np.delete(ms2, obj=np.where(ms2[:, 1] < int_threshold), axis=0)

        # if >= 10 fragments, at least 4 frags < 200
        ms2_200check = np.delete(ms2, obj=np.where(ms2[:, 0] >= 200), axis=0)
        if (ms2.shape[0] >= 10 and ms2_200check.shape[0] < 4) == True:
            continue

        ms2[:, 1] = 100 * ms2[:, 1] / max(ms2[:, 1])
        ms2[:, 1] = np.sqrt(ms2[:, 1])

        ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] < start_value), axis=0)
        ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] >= end_value), axis=0)

        if ms2.shape[0] == 0:
            continue
        for j in range(ms2.shape[0]):
            bin_position = math.floor((ms2[j, 0] - start_value) / bin_width) + 4
            df.iloc[i, bin_position] = max(ms2[j, 1], df.iloc[i, bin_position])

    # machine learning process
    # pass in data matrix
    X = dataset_passing_in(data=df.iloc[:, 4:(df.shape[1] - 1)], bw=bin_width)
    X = np.expand_dims(X, axis=2)
    y_pred = loaded_model.predict(X)

    # prediction results fill in df last column (probability)
    df = df.rename(columns={df.columns[4504]: 'prediction'})
    df.iloc[:, 4504] = y_pred.copy()

    # sort df by predicted probability
    df = df.sort_values(['prediction'], ascending=False)

    # output file (mzxml index, precursorMZ, rt, precursor_Intensity, prediction[0,1])
    output = df.iloc[:, [0, 1, 2, 3, 4504]]
    output_filename = 'output_' + mzxml_file[0:-6] + '.csv'
    
    os.chdir(output_dir)
    output.to_csv(output_filename, index=False)

    # output MGF file
    output_df = df[df.prediction > 0.5]
    index_list = output_df['mzxml_index'].astype(int).tolist()
    indices = [(x - 1) for x in index_list]
    if len(indices) > 0:
        steroid_file = file[indices]
        output_mgf = steroid_file.copy()
        for i in range(len(output_mgf)):
            output_mgf[i] = removekey(steroid_file[i],
                                      ['num', 'centroided', 'retentionTime', 'polarity', 'msLevel', 'collisionEnergy',
                                       'peaksCount', 'lowMz','highMz', 'basePeakMz', 'basePeakIntensity',
                                       'totIonCurrent', 'precursorMz','id'])
            output_mgf[i]['params'] = dict(
                [('title', steroid_file[i]['num']), ('rtinseconds', 60 * steroid_file[i]['retentionTime']),
                 ('pepmass', steroid_file[i]['precursorMz'][0]['precursorMz']), ('charge', '0+')])

        os.chdir(output_dir)
        mgf_name = 'steroid_' + mzxml_file[0:-6] + '.mgf'
        mgf.write(output_mgf, output=mgf_name, key_order=['title', 'rtinseconds', 'pepmass', 'charge'],
                  write_charges=True)

    print('File completed')
