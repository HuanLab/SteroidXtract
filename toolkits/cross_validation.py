# This script is to perform cross validation experiments using four machine learning models.
# Shipei Xing, Oct 2, 2020
# Copyright @ The University of British Columbia

# Import Libraries
import numpy as np
import pandas as pd
import math
import os
import random
import xgboost as xgb
from random import sample
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def dataset_passing_in(path):
    dataset = pd.read_csv(path)
    X = pd.DataFrame(data = dataset.iloc[:, 0:4500]).values
    y = pd.DataFrame(data = dataset.loc[:, 'class']).astype('str').values
    return X, y
       
def make_model():
    classifier = Sequential()
    classifier.add(Conv1D(32, 3, activation = 'relu', input_shape = (4500,1)))
    classifier.add(Conv1D(32, 3, activation = 'relu'))
    classifier.add(MaxPooling1D(pool_size = 2, strides = 2))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 200, activation = 'relu'))
    classifier.add(Dense(units = 200, activation = 'relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return classifier

def CNN_crossvalidation(X_train, y_train, X_test, y_test, num_epochs):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = np.asarray(y_b_train).astype(float)

    X_train = np.expand_dims(X_train, axis = 2)
    classifier = make_model()
    class_weights = {0:(X_train.shape[0]/(X_train.shape[0] - count))/2.0, 1:(X_train.shape[0]/ count)/2.0}
    classifier.fit(X_train, y_b_train, 
                   batch_size = 10, 
                   epochs = num_epochs, 
                   class_weight = class_weights)
    
    y_b_test = y_test.copy()
    y_b_test = np.where(y_test == 'steroid', 1, y_b_test)
    y_b_test = np.where(y_test == 'nonsteroid', 0, y_b_test)
    y_b_test = y_b_test.astype('int')

    X_test = np.expand_dims(X_test , axis = 2)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_b_test, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*recall*precision / (recall + precision)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    result_list = [tp, fp, tn, fn, recall, precision, f1, mcc]

    return result_list
    
def SVM_crossvalidation(X_train, y_train, X_test, y_test):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = y_b_train.astype('int')
    y_b_train = y_b_train.reshape((y_b_train.shape[0],))

    class_weights = {0:(X_train.shape[0]/(X_train.shape[0] - count))/2.0, 1:(X_train.shape[0]/ count)/2.0}

    clf = svm.SVC(kernel='rbf',class_weight=class_weights) # Radial basis function kernel, RBF kernel
    clf.fit(X_train, y_b_train)
    
    y_b_test = y_test.copy()
    y_b_test = np.where(y_test == 'steroid', 1, y_b_test)
    y_b_test = np.where(y_test == 'nonsteroid', 0, y_b_test)
    y_b_test = y_b_test.astype('int')

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_b_test, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*recall*precision / (recall + precision)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    result_list = [tp, fp, tn, fn, recall, precision, f1, mcc]

    return result_list

def RF_crossvalidation(X_train, y_train, X_test, y_test):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = y_b_train.astype('int')
    y_b_train = y_b_train.reshape((y_b_train.shape[0],))

    class_weights = {0:(X_train.shape[0]/(X_train.shape[0] - count))/2.0, 1:(X_train.shape[0]/ count)/2.0}

    clf = RandomForestClassifier(n_estimators=100,class_weight=class_weights) # Random forest
    clf.fit(X_train, y_b_train)
    
    y_b_test = y_test.copy()
    y_b_test = np.where(y_test == 'steroid', 1, y_b_test)
    y_b_test = np.where(y_test == 'nonsteroid', 0, y_b_test)
    y_b_test = y_b_test.astype('int')

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_b_test, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*recall*precision / (recall + precision)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    result_list = [tp, fp, tn, fn, recall, precision, f1, mcc]

    return result_list

def XGB_crossvalidation(X_train, y_train, X_test, y_test):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = y_b_train.astype('int')
    y_b_train = y_b_train.reshape((y_b_train.shape[0],))

    clf = xgb.XGBClassifier(objective='binary:logistic',scale_pos_weight=(X_train.shape[0] - count)/count)  # XGBoost
    clf.fit(X_train, y_b_train)
    
    y_b_test = y_test.copy()
    y_b_test = np.where(y_test == 'steroid', 1, y_b_test)
    y_b_test = np.where(y_test == 'nonsteroid', 0, y_b_test)
    y_b_test = y_b_test.astype('int')

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_b_test, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*recall*precision / (recall + precision)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    result_list = [tp, fp, tn, fn, recall, precision, f1, mcc]

    return result_list

###############Cross validation I #############
X_frag, y_frag = dataset_passing_in(path = 'E:/Subclass extraction_2020/data matrix/fragX_steroidGaussian noise_pos_50-500_top20_0.1bin_504inhouse+4752steroid+79243nonsteroid+2092blank_91847_20200719.csv')
X_ste = X_frag[:10512,]
y_ste = y_frag[:10512]
X_nonste = X_frag[10512:,]
y_nonste = y_frag[10512:]


for i in range(3):
    print('CROSS VALIDATION\n')
    print(i+1)
    if i ==0:
        random.seed(1)
    if i ==1:
        random.seed(2)
    if i ==2:
        random.seed(3)
    ste_index = sample(list(range(10512)),10512)
    nonste_index = sample(list(range(81335)),81335)
    X = np.concatenate((X_ste[ste_index],X_nonste[nonste_index]),axis=0)
    y = np.concatenate((y_ste[ste_index],y_nonste[nonste_index]),axis=0)
    skf = StratifiedKFold(n_splits=10, random_state=None)
    result_CNN = []
    result_SVM = []
    result_RF = []
    result_XGB = []
    for train_index, test_index in skf.split(X,y):
        print("Train:", train_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        result_individual_CNN = CNN_crossvalidation(X_train, y_train, X_test, y_test, 10)
        result_CNN.append(result_individual_CNN)
        print(result_individual_CNN)

        result_individual_SVM = SVM_crossvalidation(X_train, y_train, X_test, y_test)
        result_SVM.append(result_individual_SVM)

        result_individual_XGB = XGB_crossvalidation(X_train, y_train, X_test, y_test)
        result_XGB.append(result_individual_XGB)

        result_individual_RF = RF_crossvalidation(X_train, y_train, X_test, y_test)
        result_RF.append(result_individual_RF)

    result_CNN_all = pd.DataFrame(np.vstack(result_CNN))
    frag_filename = 'CNN_cvI_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_CNN_all.to_csv(frag_filename)


    result_SVM_all = pd.DataFrame(np.vstack(result_SVM))
    frag_filename = 'SVM_cvI_frag_10foldcrossvalidation_' + str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_SVM_all.to_csv(frag_filename)

    result_RF_all = pd.DataFrame(np.vstack(result_RF))
    frag_filename = 'RF_cvI_frag_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_RF_all.to_csv(frag_filename)

    result_XGB_all = pd.DataFrame(np.vstack(result_XGB))
    frag_filename = 'XGB_cvI_frag_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_XGB_all.to_csv(frag_filename)


##########Cross validation II #################
######predict novel steroids###################

X_frag, y_frag = dataset_passing_in(path = 'E:/Subclass extraction_2020/data matrix/fragX_steroidGaussian noise_pos_50-500_top20_0.1bin_504inhouse+4752steroid+79243nonsteroid+2092blank_91847_20200719.csv')

for i in range(3):
    print('CROSS VALIDATION\n')
    print(i+1)
    if i ==0:
        random.seed(1)
    if i ==1:
        random.seed(2)
    if i ==2:
        random.seed(3)
    nonste = np.array_split(sample(range(10512,91847),81335), 10)
    std_list =  np.array_split(sample(range(876),876), 10)
    ste = [None]*10
    for m in range(10):
        for n in range(std_list[m].shape[0]):
            s = 12*int(std_list[m][n]) - 11
            e = 12*int(std_list[m][n])
            if n == 0:
                ste[m] = list(range(s,e+1))
            if n > 0:
                ste[m] = ste[m]+ list(range(s,e+1))

    result_CNN = []
    result_SVM = []
    result_RF = []
    result_XGB = []
    for m in range(10):

        test_index = ste[m] + list(nonste[m])
        test_index.sort()
        a = list(range(91847))
        train_index = [x for x in a if x not in test_index]

        print(m)
        X_train, X_test = X_frag[train_index], X_frag[test_index]
        y_train, y_test = y_frag[train_index], y_frag[test_index]

        result_individual_CNN = CNN_crossvalidation(X_train, y_train, X_test, y_test, 10)
        result_CNN.append(result_individual_CNN)

        result_individual_SVM = SVM_crossvalidation(X_train, y_train, X_test, y_test)
        result_SVM.append(result_individual_SVM)

        result_individual_RF = RF_crossvalidation(X_train, y_train, X_test, y_test)
        result_RF.append(result_individual_RF)

        result_individual_XGB = XGB_crossvalidation(X_train, y_train, X_test, y_test)
        result_XGB.append(result_individual_XGB)

    result_CNN_all = pd.DataFrame(np.vstack(result_CNN))
    frag_filename = 'CNN_cvII_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_CNN_all.to_csv(frag_filename)

    result_SVM_all = pd.DataFrame(np.vstack(result_SVM))
    frag_filename = 'SVM_cvII_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_SVM_all.to_csv(frag_filename)

    result_RF_all = pd.DataFrame(np.vstack(result_RF))
    frag_filename = 'RF_cvII_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_RF_all.to_csv(frag_filename)

    result_XGB_all = pd.DataFrame(np.vstack(result_XGB))
    frag_filename = 'XGB_cvII_10foldcrossvalidation_'+ str(i+1) + '.csv'
    os.chdir('E:/Subclass extraction_2020/cross validaton')
    result_XGB_all.to_csv(frag_filename)
