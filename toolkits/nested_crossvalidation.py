# nested cross validation
# outer loop: unbiased model performance assessment, for algorithm selection
#             outer k = 5
# inner loop: hyperparameter tuning
#             inner k = 3

# Import Library
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
from sklearn.ensemble import RandomForestClassifier

def dataset_passing_in(path):
    dataset = pd.read_csv(path)
    dataset.rename(columns = {'X1501': 'class'}, inplace = True)
    X = pd.DataFrame(data = dataset.iloc[:, 0:1500]).values
    y = pd.DataFrame(data = dataset.loc[:, 'class']).astype('str').values
    return X, y
       
def make_model(unitNo = 400):
    classifier = Sequential()
    classifier.add(Conv1D(32, 3, activation = 'relu', input_shape = (1500,1)))
    classifier.add(Conv1D(32, 3, activation = 'relu'))
    classifier.add(MaxPooling1D(pool_size = 2, strides = 2))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = unitNo, activation = 'relu'))
    classifier.add(Dense(units = unitNo, activation = 'relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return classifier

def CNN_crossvalidation(X_train, y_train, X_test, y_test, num_epochs=10, unitNo = 200):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = np.asarray(y_b_train).astype(float)

    X_train = np.expand_dims(X_train, axis = 2)
    classifier = make_model(unitNo = unitNo)
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
    

def RF_crossvalidation(X_train, y_train, X_test, y_test, treeNo = 100):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = y_b_train.astype('int')
    y_b_train = y_b_train.reshape((y_b_train.shape[0],))

    class_weights = {0:(X_train.shape[0]/(X_train.shape[0] - count))/2.0, 1:(X_train.shape[0]/ count)/2.0}

    clf = RandomForestClassifier(n_estimators=treeNo,class_weight=class_weights) # Random forest
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

def XGB_crossvalidation(X_train, y_train, X_test, y_test, treeNo = 100):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = y_b_train.astype('int')
    y_b_train = y_b_train.reshape((y_b_train.shape[0],))

    clf = xgb.XGBClassifier(objective='binary:logistic',scale_pos_weight=(X_train.shape[0] - count)/count,
                            n_estimators= treeNo)  # XGBoost
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

def CNN_model(X, y, num_epochs):
    y_b = y.copy()
    y_b = np.where(y == 'steroid', 1, y_b)
    y_b = np.where(y == 'nonsteroid', 0, y_b)
    count = (y == 'steroid').sum()
    y_b = np.asarray(y_b).astype(float)

    X = np.expand_dims(X, axis = 2)
    classifier = make_model()
    class_weights = {0:(X.shape[0]/(X.shape[0] - count))/2.0, 1:(X.shape[0]/ count)/2.0}
    classifier.fit(X, y_b, 
                   batch_size = 10, 
                   epochs = num_epochs, 
                   class_weight = class_weights)
    
    y_pred = classifier.predict(X)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_b, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*recall*precision / (recall + precision)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    result_list = [tp, fp, tn, fn, recall, precision, f1, mcc]

    os.chdir('E:/Subclass extraction_2020/Final codes & file templates')
    json = classifier.to_json()
    with open('model_200_0729.json', 'w') as json_file:
        json_file.write(json)
    classifier.save_weights('model_200_0729.h5')

    return result_list
 
## 20210119 cross validation ######
##### nested CV I #########
X_frag, y_frag = dataset_passing_in(path = 'E:/SteroidXtract_2020/data matrix/fragX_pos_50-200_0.1bin_30176steroid_106650nonsteroid_noiselevel0-50_20210119.csv')
X_ste = X_frag[:30176,]
y_ste = y_frag[:30176]
X_nonste = X_frag[30176:,]
y_nonste = y_frag[30176:]

random.seed(1)
ste_index = sample(list(range(30176)), 30176)
nonste_index = sample(list(range(106650)), 106650)
X = np.concatenate((X_ste[ste_index], X_nonste[nonste_index]), axis=0)
y = np.concatenate((y_ste[ste_index], y_nonste[nonste_index]), axis=0)

skf = StratifiedKFold(n_splits = 5 , random_state=None)
innerskf = StratifiedKFold(n_splits = 3 , random_state=None)

##### CNN
result_CNN = []
for train_index, test_index in skf.split(X, y):
    print("Train:", train_index)
    X_outtrain, X_outtest = X[train_index], X[test_index]
    y_outtrain, y_outtest = y[train_index], y[test_index]

    h = 0
    inner_result = []
    unitNo = 100
    for intrain_index, intest_index in innerskf.split(X_outtrain, y_outtrain):
        print('h = ' + str(h))
        X_intrain, X_intest = X_outtrain[intrain_index], X_outtrain[intest_index]
        y_intrain, y_intest = y_outtrain[intrain_index], y_outtrain[intest_index]
        if h==0:
            unitNo = 100
        if h==1:
            unitNo = 200
        if h==2:
            unitNo = 400
        result = CNN_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                num_epochs=10, unitNo=unitNo)[7]
        h = h + 1
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        unitNo = 100
    if index == 1:
        unitNo = 200
    if index == 2:
        unitNo = 400

    result_individual_CNN = CNN_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        num_epochs=10, unitNo=unitNo)
    result_CNN.append(result_individual_CNN)
    print(result_individual_CNN)

result_CNN_all = pd.DataFrame(np.vstack(result_CNN))
filename = 'CNN_cvI_epoch10_20210120.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_I_20210120')
result_CNN_all.to_csv(filename)

####### RF
result_RF = []
k = 0
for train_index, test_index in skf.split(X, y):
    print('k = ' + str(k))
    print("Train:", train_index)
    X_outtrain, X_outtest = X[train_index], X[test_index]
    y_outtrain, y_outtest = y[train_index], y[test_index]

    h = 0
    inner_result = []
    treeNo = 100
    for intrain_index, intest_index in innerskf.split(X_outtrain, y_outtrain):
        print('h = ' + str(h))
        X_intrain, X_intest = X_outtrain[intrain_index], X_outtrain[intest_index]
        y_intrain, y_intest = y_outtrain[intrain_index], y_outtrain[intest_index]
        if h==0:
            treeNo = 100
        if h==1:
            treeNo = 200
        if h==2:
            treeNo = 400
        result = RF_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                treeNo= treeNo)[7]
        h = h + 1
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        treeNo = 100
    if index == 1:
        treeNo = 200
    if index == 2:
        treeNo = 400

    result_individual_RF = RF_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        treeNo= treeNo)
    result_RF.append(result_individual_RF)
    print(result_individual_RF)
    k = k + 1

result_RF_all = pd.DataFrame(np.vstack(result_RF))
filename = 'RF_cvI_epoch10_20210120.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_I_20210120')
result_RF_all.to_csv(filename)

####### XGB
result_XGB = []
k = 0
for train_index, test_index in skf.split(X, y):
    print('k = ' + str(k))
    print("Train:", train_index)
    X_outtrain, X_outtest = X[train_index], X[test_index]
    y_outtrain, y_outtest = y[train_index], y[test_index]

    h = 0
    inner_result = []
    treeNo = 100
    for intrain_index, intest_index in innerskf.split(X_outtrain, y_outtrain):
        print('h = ' + str(h))
        X_intrain, X_intest = X_outtrain[intrain_index], X_outtrain[intest_index]
        y_intrain, y_intest = y_outtrain[intrain_index], y_outtrain[intest_index]
        if h==0:
            treeNo = 100
        if h==1:
            treeNo = 200
        if h==2:
            treeNo = 400
        result = XGB_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                treeNo= treeNo)[7]
        h = h + 1
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        treeNo = 100
    if index == 1:
        treeNo = 200
    if index == 2:
        treeNo = 400

    result_individual_XGB = XGB_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        treeNo= treeNo)
    result_XGB.append(result_individual_XGB)
    print(result_individual_XGB)
    k = k + 1

result_XGB_all = pd.DataFrame(np.vstack(result_XGB))
filename = 'XGB_cvI_epoch10_20210120.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_I_20210120')
result_XGB_all.to_csv(filename)


##########20210121 cross validation#######
###### nested CN II ###################
outer_index = pd.read_csv('E:/SteroidXtract_2020/data matrix/nested_CVII_outerloop_20210121.csv')
inner_index = pd.read_csv('E:/SteroidXtract_2020/data matrix/nested_CVII_innerloop_20210122.csv')

for i in range(5):
    globals()['outer'+str(i)] = list(map(int, outer_index.iloc[i,0].split(";")))

for i in range(5):
    for j in range(3):
        globals()['inner'+str(j)+str(i)] = list(map(int, inner_index.iloc[j, i].split(";")))


# inner00, inner10, inner20
# inner01, inner11, inner21
# inner02, inner12, inner22
# inner03, inner13, inner23
# inner04, inner14, inner24

##### CNN
result_CNN = []
for i in range(5):
    print('i = ' + str(i))
    test_index = [(x-1) for x in eval('outer'+str(i))]
    train_index = [x for x in range(136826) if not x in test_index]
    print("Train:", train_index)
    X_outtrain, X_outtest = X_frag[train_index], X_frag[test_index]
    y_outtrain, y_outtest = y_frag[train_index], y_frag[test_index]

    inner_result = []
    unitNo = 100
    for j in range(3):
        print('j = ' + str(j))
        intest_index =  [(x-1) for x in eval('inner'+str(j)+str(i))]
        intrain_index = [x for x in train_index if not x in intest_index]
        X_intrain, X_intest = X_frag[intrain_index], X_frag[intest_index]
        y_intrain, y_intest = y_frag[intrain_index], y_frag[intest_index]
        if j==0:
            unitNo = 100
        if j==1:
            unitNo = 200
        if j==2:
            unitNo = 400
        result = CNN_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                num_epochs=10, unitNo=unitNo)[7]
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        unitNo = 100
    if index == 1:
        unitNo = 200
    if index == 2:
        unitNo = 400
    print('i = ' + str(i))
    print('unitNo' + str(unitNo))

    result_individual_CNN = CNN_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        num_epochs=10, unitNo=unitNo)
    result_CNN.append(result_individual_CNN)
    print(result_individual_CNN)
result_CNN_all = pd.DataFrame(np.vstack(result_CNN))
filename = 'CNN_cvII_epoch10_20210122.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_II_20210122')
result_CNN_all.to_csv(filename)

##### RF
result_RF = []
for i in range(5):
    print('i = ' + str(i))
    test_index = [(x-1) for x in eval('outer'+str(i))]
    train_index = [x for x in range(136826) if not x in test_index]
    print("Train:", train_index)
    X_outtrain, X_outtest = X_frag[train_index], X_frag[test_index]
    y_outtrain, y_outtest = y_frag[train_index], y_frag[test_index]

    inner_result = []
    treeNo = 100
    for j in range(3):
        print('j = ' + str(j))
        intest_index =  [(x-1) for x in eval('inner'+str(j)+str(i))]
        intrain_index = [x for x in train_index if not x in intest_index]
        X_intrain, X_intest = X_frag[intrain_index], X_frag[intest_index]
        y_intrain, y_intest = y_frag[intrain_index], y_frag[intest_index]
        if j==0:
            treeNo = 100
        if j==1:
            treeNo = 200
        if j==2:
            treeNo = 400
        result = RF_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                treeNo=treeNo)[7]
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        treeNo = 100
    if index == 1:
        treeNo = 200
    if index == 2:
        treeNo = 400
    print('i = ' + str(i))
    print('j = ' + str(j))
    print('treeNo' + str(treeNo))

    result_individual_RF = RF_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        treeNo=treeNo)
    result_RF.append(result_individual_RF)
    print(result_individual_RF)
result_RF_all = pd.DataFrame(np.vstack(result_RF))
filename = 'RF_cvII_epoch10_20210122.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_II_20210122')
result_RF_all.to_csv(filename)

##### XGB
result_XGB = []
for i in range(5):
    print('i = ' + str(i))
    test_index = [(x-1) for x in eval('outer'+str(i))]
    train_index = [x for x in range(136826) if not x in test_index]
    print("Train:", train_index)
    X_outtrain, X_outtest = X_frag[train_index], X_frag[test_index]
    y_outtrain, y_outtest = y_frag[train_index], y_frag[test_index]

    inner_result = []
    treeNo = 100
    for j in range(3):
        print('j = ' + str(j))
        intest_index =  [(x-1) for x in eval('inner'+str(j)+str(i))]
        intrain_index = [x for x in train_index if not x in intest_index]
        X_intrain, X_intest = X_frag[intrain_index], X_frag[intest_index]
        y_intrain, y_intest = y_frag[intrain_index], y_frag[intest_index]
        if j==0:
            treeNo = 100
        if j==1:
            treeNo = 200
        if j==2:
            treeNo = 400
        result = XGB_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                treeNo=treeNo)[7]
        inner_result.append(result)

    index = inner_result.index(max(inner_result))
    if index == 0:
        treeNo = 100
    if index == 1:
        treeNo = 200
    if index == 2:
        treeNo = 400
    print('i = ' + str(i))
    print('j = ' + str(j))
    print('treeNo' + str(treeNo))

    result_individual_XGB = XGB_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        treeNo=treeNo)
    result_XGB.append(result_individual_XGB)
    print(result_individual_XGB)
result_XGB_all = pd.DataFrame(np.vstack(result_XGB))
filename = 'XGB_cvII_epoch10_20210122.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_II_20210122')
result_XGB_all.to_csv(filename)


## 20210124 unaugmented data matrix ######
##### nested CV I #########
X_frag, y_frag = dataset_passing_in(path = 'E:/SteroidXtract_2020/data matrix/unaug_fragX_pos_50-200_0.1bin_2575steroid_53325nonsteroid_20210124.csv')
X_ste = X_frag[:2575,]
y_ste = y_frag[:2575]
X_nonste = X_frag[2575:,]
y_nonste = y_frag[2575:]

random.seed(1)
ste_index = sample(list(range(2575)), 2575)
nonste_index = sample(list(range(53325)), 53325)
X = np.concatenate((X_ste[ste_index], X_nonste[nonste_index]), axis=0)
y = np.concatenate((y_ste[ste_index], y_nonste[nonste_index]), axis=0)

skf = StratifiedKFold(n_splits = 5 , random_state=None)
innerskf = StratifiedKFold(n_splits = 3 , random_state=None)

##### CNN
result_CNN = []
for train_index, test_index in skf.split(X, y):
    print("Train:", train_index)
    X_outtrain, X_outtest = X[train_index], X[test_index]
    y_outtrain, y_outtest = y[train_index], y[test_index]

    h = 0
    inner_result = []
    unitNo = 100
    for intrain_index, intest_index in innerskf.split(X_outtrain, y_outtrain):
        print('h = ' + str(h))
        X_intrain, X_intest = X_outtrain[intrain_index], X_outtrain[intest_index]
        y_intrain, y_intest = y_outtrain[intrain_index], y_outtrain[intest_index]
        if h==0:
            unitNo = 100
        if h==1:
            unitNo = 200
        if h==2:
            unitNo = 400
        result = CNN_crossvalidation(X_intrain, y_intrain, X_intest, y_intest,
                                                num_epochs=10, unitNo=unitNo)[7]
        h = h + 1
        inner_result.append(result)
        print('h = ' + str(h))
        print('unitNo' + str(unitNo))

    index = inner_result.index(max(inner_result))
    if index == 0:
        unitNo = 100
    if index == 1:
        unitNo = 200
    if index == 2:
        unitNo = 400

    result_individual_CNN = CNN_crossvalidation(X_outtrain, y_outtrain, X_outtest, y_outtest,
                                        num_epochs=10, unitNo=unitNo)
    result_CNN.append(result_individual_CNN)
    print(result_individual_CNN)


result_CNN_all = pd.DataFrame(np.vstack(result_CNN))
filename = 'unaug_CNN_cvI_epoch10_20210124.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_I_20210120')
result_CNN_all.to_csv(filename)


# make the final CNN model
X, y = dataset_passing_in(path = 'X:/Users/Shipei_Xing/SteroidXtract/training/fragX_pos_50-200_0.1bin_30176steroid_106650nonsteroid_noiselevel0-50_20210119.csv')
result = CNN_model(X, y ,num_epochs=10)


# compound classes that are misclassified by the model

def CNN_cv(X_train, y_train, X_test, y_test):
    y_b_train = y_train.copy()
    y_b_train = np.where(y_train == 'steroid', 1, y_b_train)
    y_b_train = np.where(y_train == 'nonsteroid', 0, y_b_train)
    count = (y_train == 'steroid').sum()
    y_b_train = np.asarray(y_b_train).astype(float)

    X_train = np.expand_dims(X_train, axis = 2)
    classifier = make_model(unitNo = 400)
    class_weights = {0:(X_train.shape[0]/(X_train.shape[0] - count))/2.0, 1:(X_train.shape[0]/ count)/2.0}
    classifier.fit(X_train, y_b_train,
                   batch_size = 10,
                   epochs = 10,
                   class_weight = class_weights)

    y_b_test = y_test.copy()
    y_b_test = np.where(y_test == 'steroid', 1, y_b_test)
    y_b_test = np.where(y_test == 'nonsteroid', 0, y_b_test)
    y_b_test = y_b_test.astype('int')

    X_test = np.expand_dims(X_test, axis=2)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    index = np.where((y_predx == y_b_test) == False)[0]
    result_list = index.tolist()

    return result_list

##### CNN
result = []
for i in range(5):
    print('i = ' + str(i))
    test_index = [(x-1) for x in eval('outer'+str(i))]
    train_index = [x for x in range(136826) if not x in test_index]
    print("Train:", train_index)
    X_train, X_test = X_frag[train_index], X_frag[test_index]
    y_train, y_test = y_frag[train_index], y_frag[test_index]

    result_individual = CNN_cv(X_train, y_train, X_test, y_test)
    result.append(result_individual)

    print(result.shape)

result_all = pd.DataFrame(np.vstack(result))
filename = 'CNN_cvII_misclassified_indices_20210127.csv'
os.chdir('E:/SteroidXtract_2020/cross validaton/nested_CV_II_20210122')
result_all.to_csv(filename)
