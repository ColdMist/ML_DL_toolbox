import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
#from keras.utils import np_utils
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def build_tf_preprocessing(input_data):

    inputs = {}
    for name, column in input_data.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(input_data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = layers.StringLookup(vocabulary=np.unique(input_data[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    data_features_dict = {name: np.array(value)
                             for name, value in input_data.items()}

    return inputs

def build_model(input = None, output = None):
  if (input != None and output != None):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input,)),
        keras.layers.Dense(120, activation=tf.nn.relu,  kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(64, activation = tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(output, activation=tf.nn.sigmoid),
    ])
  else:
    print('input and output shape is not defined')
    return None
  #return model
  opt = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
  )
  model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
  return model

def build_sklearn_model(model_name = 'random_forest'):
    cls = None
    if model_name == 'random_forest':
        cls = RandomForestClassifier(max_depth=500, random_state=0)
    elif model_name == 'MLP':
        cls = MLPClassifier()
    elif model_name == 'k_neighbor_classifier':
        cls = KNeighborsClassifier(n_neighbors=2)
    elif model_name == 'svm':
        cls = svm.SVC(kernel='rbf', gamma=0.1)
    elif model_name == 'gnb':
        cls = GaussianNB()
    elif model_name == 'd_tree':
        cls = DecisionTreeClassifier()
    elif model_name == 'logistic_regression':
        cls = LogisticRegression()
    else:
        print('model name not supported yet!')
        exit()
    return cls

def fit_sklearn_model(model, training_data):
    return model.fit(training_data[0], training_data[1])

def predict_on_sklearn_trained_model(trained_model, test_data):
    return trained_model.predict(test_data)