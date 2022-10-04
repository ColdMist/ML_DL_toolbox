import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
# Make numpy values easier to read.
#np.set_printoptions(precision=3, suppress=True)
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
#from tensorflow.keras import layers
from utils.helper_functions import *
from utils.XAI_functions import *
from utils.model_utils import *

if __name__ == '__main__':
    #TODO build argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  type=str, required=True)
    parser.add_argument('--label',  type=str, required=True)
    parser.add_argument('--dataset_seperator', type=str, default=',')
    parser.add_argument('--data_header', action='store_true')
    #parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--training_procedures',  nargs='+', required= True)
    parser.add_argument('--sklearn_model_name', type=str)
    parser.add_argument('--xai_sklearn', action='store_true')
    parser.add_argument('--obtain_feat_imp', action='store_true')
    parser.add_argument('--xai_show_feat_imp', action='store_true')
    parser.add_argument('--DNN_tf_imp_feat', action='store_true')
    ######
    # dataset = 'Persistent_vs_NonPersistent.csv'
    # label = 'Persistency_Flag'
    # data_path = os.path.join('dataset/', dataset)
    # dataset_seperator = ','
    # data_header = True
    # training_procedures = ['sklearn_ml_libraries', 'DNN_tf']
    # perform_XAI_on_sklearn = True
    # obtain_feature_importance = True
    # show_important_features_XAI = True
    # DNN_tf_important_features = True
    args = parser.parse_args()
    dataset = args.dataset
    label = args.label
    data_path = os.path.join('dataset/', dataset)
    dataset_seperator = args.dataset_seperator
    data_header = args.data_header
    training_procedures = args.training_procedures
    sklearn_model_name = args.sklearn_model_name
    perform_XAI_on_sklearn = args.xai_sklearn
    obtain_feature_importance = args.obtain_feat_imp
    show_important_features_XAI = args.xai_show_feat_imp
    DNN_tf_important_features = args.DNN_tf_imp_feat

    important_features = None

    if data_header == True:
        df = pd.read_table(data_path, sep=dataset_seperator)
    else:
        df = pd.read_table(data_path, sep=dataset_seperator, header=None)

    cat_dict, data = process_data_for_ml(df)
    X = data.drop(label, axis=1)  # independent Feature
    y = data[label]  # dependent Feature

    X_train, X_test, y_train, y_test = create_split(X,y, portion=0.2)

    if 'sklearn_ml_libraries' in training_procedures:
        model = build_sklearn_model(sklearn_model_name)
        fitted_model = fit_sklearn_model(model, training_data=[X_train,y_train])
        prediction_test = predict_on_sklearn_trained_model(trained_model=fitted_model, test_data=X_test)

        TP, FP, TN, FN, f1 = calculate_report(y_test, prediction_test)
        print('classification report sklearn: ')
        print(f"The report on test result is: \n"
              f"True positive: {TP}, False positive: {FP}, True negative: {TN}, False negative: {FN} and F1: {f1}")
        #print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_pred_test)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_test, prediction_test):.4f}")
        if obtain_feature_importance == True:
            model = build_sklearn_model('random_forest')
            fitted_model =fit_sklearn_model(model, training_data=[X_train,y_train])
            df_fi = feature_imp(X_test, model)[:]
            df_fi.set_index('feature', inplace=True)
            df_fi.plot(kind='barh')
            plt.title('Feature Importance according to RFC')
            plt.savefig("saved_figures/feature_imp.pdf", format='pdf', dpi=1000, bbox_inches='tight')
            #plt.show()
        if perform_XAI_on_sklearn == True:
            explainer, shap_values = build_explainer_on_classifier(model, X_test)
            wrong_prediction = [i for i in range(len(y_test.values)) if y_test.values[i] != prediction_test[i]]
            correct_prediction = [i for i in range(len(y_test.values)) if y_test.values[i] == prediction_test[i]]
            shap.summary_plot(shap_values[1][correct_prediction, :], X_test.values[correct_prediction, :],
                              feature_names=X_test.columns, plot_size=[14, 6], show=False)
            plt.savefig("saved_figures/summary_plot.pdf", format='pdf', dpi=1000, bbox_inches='tight')
            #plt.show()
            if show_important_features_XAI == True:
                correctly_predicted_test = X_test.values[correct_prediction, :]
                correctly_predicted_test_df = pd.DataFrame(correctly_predicted_test, columns=X_test.columns)

                explainer_correct_prediction, shap_values_correctly_predicted = build_explainer_on_classifier(model,
                                                                                                              correctly_predicted_test_df)
                vals = np.abs(shap_values_correctly_predicted).mean(0)

                feature_importance = pd.DataFrame(list(zip(correctly_predicted_test_df.columns, sum(vals))),
                                                  columns=['col_name', 'feature_importance_vals'])
                feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                n=20
                important_features = feature_importance.head(n)['col_name'].unique()
                print('the most important features are sorted: ', important_features)

    if 'DNN_tf' in training_procedures:
        input_features = X_train.shape[1]
        out_features = len(np.unique(y_train))
        if DNN_tf_important_features & show_important_features_XAI:
            X_train = X_train[important_features]
            X_test = X_test[important_features]
        model_tf = build_model_tf(input=X_train.shape[1], output=1)
        model_tf.fit(X_train, y_train, validation_split=0.1, epochs=500, batch_size=128, verbose=2)
        prediction_test_tf = model_tf.predict(X_test)
        prediction_test_tf = [1 if i[0]>=0.5 else 0 for i in prediction_test_tf]
        #print(prediction_test_tf)
        print('classification report tf: ')
        TP, FP, TN, FN, f1 = calculate_report(y_test, prediction_test_tf)
        print(f"The report on test result is: \n"
              f"True positive: {TP}, False positive: {FP}, True negative: {TN}, False negative: {FN} and F1: {f1}")
        # print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_pred_test)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_test, prediction_test_tf):.4f}")
        test_loss, test_acc = model_tf.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)