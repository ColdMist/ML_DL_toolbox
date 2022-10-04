import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
#from tensorflow.keras import layers
from utils.helper_functions import *
from utils.XAI_functions import *
from utils.model_utils import *

if __name__ == '__main__':
    #TODO build argparse:

    ######
    dataset = 'Persistent_vs_NonPersistent.csv'
    label = 'Persistency_Flag'
    data_path = os.path.join('dataset/', dataset)
    dataset_seperator = ','
    data_header = True
    training_procedure = 'sklearn_ml_libraries'
    perform_XAI_on_sklearn = True
    obtain_feature_importance = True

    if data_header == True:
        df = pd.read_table(data_path, sep=dataset_seperator)
    else:
        df = pd.read_table(data_path, sep=dataset_seperator, header=None)

    cat_dict, data = process_data_for_ml(df)
    X = data.drop(label, axis=1)  # independent Feature
    y = data[label]  # dependent Feature

    X_train, X_test, y_train, y_test = create_split(X,y, portion=0.2)

    if training_procedure == 'sklearn_ml_libraries':
        model = build_sklearn_model('random_forest')
        fitted_model = fit_sklearn_model(model, training_data=[X_train,y_train])
        prediction_test = predict_on_sklearn_trained_model(trained_model=fitted_model, test_data=X_test)

        TP, FP, TN, FN, f1 = calculate_report(y_test, prediction_test)
        print(f"The report on test result is: \n"
              f"True positive: {TP}, False positive: {FP}, True negative: {TN}, False negative: {FN} and F1: {f1}")
        #print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_pred_test)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_test, prediction_test):.4f}")
        if obtain_feature_importance == True:
            #TODO fix feature importance
        if perform_XAI_on_sklearn == True:
            explainer, shap_values = build_explainer_on_classifier(model, X_test)
