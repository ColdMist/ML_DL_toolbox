import pandas as pd
import shap
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)

def feature_imp(data, model):
    #TODO fix feature importance
    fi = pd.DataFrame()
    fi["feature"] = data.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=True)

def set_column_name(df, cols):
    df.columns = cols
    return df

def process_data_for_ml(data):
    #Input: data in tabular format
    #Output: categorical variable mapping and the data
    column_names = data.columns
    numerical_features = []
    categorical_features = []

    actual_categorical_variable = []
    actual_numeric_variable_with_noise = []
    actual_numeric_variable_without_noise = []

    for col in column_names:
        try:
            data[col] = data[col].astype(float)
            numerical_features.append(col)
        except:
            categorical_features.append(col)

    for var in categorical_features:
        numeric_value_count = 0
        categorical_value_count = 0

        for i in data.index:
            try:
                float(data.loc[i, var])
                #print('here')
                numeric_value_count+=1
            except:
                categorical_value_count+=1
        total_data = len(data[var])
        categorical_value_count_percentage = (categorical_value_count / total_data) * 100
        numeric_value_count_percentage = (numeric_value_count / total_data) * 100
        if numeric_value_count_percentage == 100:
            actual_numeric_variable_without_noise.append(var)

        elif numeric_value_count_percentage < 100 and numeric_value_count_percentage >= 90:
            actual_numeric_variable_with_noise.append(var)
        else:
            actual_categorical_variable.append(var)
    numerical_variables = list(set(column_names) - set(actual_categorical_variable))
    #print(numerical_variables)
    label_encoders = dict()
    categorical_mapping = dict()
    for var in actual_categorical_variable:
        label_encoders[var] = preprocessing.LabelEncoder()
        data[var] = label_encoders[var].fit_transform(data[var])
        #print('after_transformation', data[var])
        #print('before_transformation', label_encoders[var].inverse_transform(data[var]))
        #categorical_mapping[var] = dict(zip(label_encoders[var].classes_, label_encoders[var].transform(label_encoders[var].classes_)))
        categorical_mapping[var] = dict(zip(label_encoders[var].transform(label_encoders[var].classes_), label_encoders[var].classes_))
        #print('#################################')
    return categorical_mapping, data

def change_to_original(X, cat_dict):
  #print(X_test['BusinessTravel'])
  X_original = X.copy()
  #cat_dict['Attrition'], inplace=True)
  for k in cat_dict.keys():
      #print(cat_dict[k])
      try:
        X_original[k] = X_original[k].map(cat_dict[k])
      except:
        continue
  #print(X_test_original)
  return X_original


def create_split(X,y, portion = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=portion, random_state=42)
    return X_train, X_test, y_train, y_test

def plot_confusion_metrics(cm, classes):
    classes_str = ['class-' + str(i) for i in classes]
    df_cm = pd.DataFrame(cm, index=[i for i in classes_str],
                         columns=[i for i in classes_str])
    plt.title('Figure')
    sns.set(font_scale=3)
    #plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    sns.set(font_scale=2)
    #pp_matrix(df_cm, cmap='YlGnBu')
    #pp_matrix(df_cm)
    plt.show()

def calculate_report(y_true, y_pred):
    print('####################')
    '''obtained from solution of
    https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal'''
    cm = confusion_matrix(y_true, y_pred)
    unique_classes = np.unique(list(y_pred) + list(y_true))
    print(unique_classes)
    plot_confusion_metrics(cm, unique_classes)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm[:].sum() - (FP + FN + TP)

    f1 = f1_score(y_true, y_pred, average='weighted')
    return TP, FP, TN, FN, f1

