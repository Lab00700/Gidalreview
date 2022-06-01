import gc
import sqlite3
import pandas as pd
import numpy as np
import os
from multiprocessing import Process, Queue

import warnings
warnings.filterwarnings('ignore')

from daal4py.sklearn import patch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import joblib

from sklearn.metrics import confusion_matrix, classification_report
from AutoML import AutoML

database_list = ['crowling_all',
                 'non_del',
                 'non_pic',
                 'non_del_pic']

def rfc(X_train,X_test,y_train,y_test):
    n_estimators = list(range(1, 1024, 128))
    max_depth = list(range(1, 20, 4))
    max_depth.insert(0, None)
    param = {
        'n_estimators':[n_estimators,128],
        'criterion': ['gini', 'entropy'],
        'max_depth': [max_depth, 4]
    }
    model = RandomForestClassifier(random_state=42)
    temp = AutoML(model, param_grid=param)
    result, score, dict = temp.fit(X_test,y_test,X_train,y_train)
    y_pred = temp.predict(X_test,X_train,y_train)
    return score, dict, y_pred, temp

def _svm(X_train,X_test,y_train,y_test):
    C = list(range(1, 16, 4))
    gamma = list(range(1, 64, 16))
    param = {
        'C': [C,4],
        'kernel':['linear','rbf'],
        'gamma':[gamma,16]

    }
    model = SVC(random_state=42)
    temp = AutoML(model, param_grid=param)
    result, score, dict = temp.fit(X_test,y_test,X_train,y_train)
    y_pred=temp.predict(X_test,X_train,y_train)
    return score, dict, y_pred, temp
    
def fold(q,X_train,X_test,y_train,y_test):
    best=0
    best_r=0

    result = _svm(X_train, X_test, y_train, y_test)
    print("\n{} SVM : {}\n".format(os.getpid(),result[:2]))
    r = [True for v in range(len(result[2])) if
         list(result[2] == y_test)[v] or
         list(result[2] == y_test + 1)[v] or
         list(result[2] == y_test - 1)[v]]
    r = sum(r) / len(result[2])
    print("{} SVM Reliable ±1 : {}\n".format(os.getpid(), r))
    matrix = confusion_matrix(result[2], y_test)
    print(matrix)
    print(result[3])
    pred=[]
    if result[0]>best:
        best=result[0]
        best_r=r
        model_name='SVM'
        dict=result[1]
        pred.append(result[3])
        pred.append(X_train)
        pred.append(y_train)


    return_value=[]
    return_value.append(best)
    return_value.append(best_r)
    return_value.append(model_name)
    return_value.append(dict)
    return_value.append(pred)
    q.put(return_value)


if __name__ == '__main__':
    sql_ = sqlite3.connect('learn_data.db')  # 있으면 파일 불러오기
    table_name = 'Learn_Data'
    print("Learning DB File Loading...\n")
    df = pd.DataFrame()
    table_value = 0

    while True:
        try:
            sql_to_df = pd.read_sql("SELECT * FROM " + str(table_name) + str(table_value), sql_, index_col=None)
            sql_to_df_32 = sql_to_df.astype('float32')
            df = pd.concat([df, sql_to_df_32], axis=1)
            print("Learning DB Table {} Loading.".format(table_value))
            table_value = table_value + 1
            del [sql_to_df, sql_to_df_32]
            gc.collect()

        except:
            print("Learning DB File Loading is Done.")
            break

    core = 4
    kfold = StratifiedKFold(n_splits=core)

    value_count=df['total'].value_counts()
    min_count=min(value_count)
    min_log=np.log10(min_count)

    print(value_count)

    data=df.copy()

    del df
    gc.collect()

    label=data['total'].astype('int8')
    data=data.drop(['total'],axis=1)
    print(label.value_counts())

    print("\nStart Learning...\n")
    model=None
    for train_index, test_index in kfold.split(X=data,y=label):
        X_train,X_test=data.loc[train_index],data.loc[test_index]
        y_train,y_test=label.loc[train_index],label.loc[test_index]

        best_predict=fold(X_train,X_test,y_train,y_test)
        del [X_train, X_test, y_train, y_test]
        gc.collect()

    del [data, label]
    gc.collect()

    print(best_predict[0])
    print(best_predict[1])
    print(best_predict[2])
    print(best_predict[3])
    model=best_predict[4][0].best_es
    print(model)
    model.fit(best_predict[4][1], best_predict[4][2])
    joblib.dump(model, './best_model.pkl')
