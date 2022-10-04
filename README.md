# ML_DL_toolbox [under construction]:
Python toolbox for evaluating ML/DL datasets. Enables auto data preperation for ML and DL and fit into various models.

# Available modules:
1. Scikit_learn: Random Forest, Decision Tree, K Nearest Neighbor, SVM etc
2. Deep Learning Model: based on tensorflow [adjustable layers coming..]
3. XAI analysis using shap: summary plot, important features etc

# How to run:
1. mkdir dataset and put the dataset inside the 'dataset' folder.
2. mkdir saved_figures where figures will be saved
3. run python script.py using command line arguments. --dataset [dataset_name] --label [target_column] --dataset_seperator [, or | or ;] --data_header [true/false whether you have headers on dataset] --training_procedures [current_available training procedures(see example command)] --xai_sklearn [true/false depending on if one wants xai on ml data]

# Example command:
python script.py --dataset Persistent_vs_NonPersistent.csv --label Persistency_Flag --dataset_seperator , --data_header --training_procedures sklearn_ml_libraries DNN_tf --xai_sklearn --obtain_feat_imp --xai_show_feat_imp --DNN_tf_imp_feat


