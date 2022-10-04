import shap
import numpy as np
import pandas as pd

def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance

def build_explainer_on_classifier(classifier, data, explainer = 'Tree'):
    shap_values = None
    if explainer == 'Tree':
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(data)
    else:
        pass
    return explainer, shap_values

def build_explainer_on_KNN(cls, X):
    #f = lambda x: cls.predict_proba(x)[:, 1]
    #med = X.median().values.reshape((1, X.shape[1]))
    #print(X)
    #exit()
    #X_summarized = shap.kmeans(X, k=3)
    X_summary = shap.kmeans(X, 10)
    #print(X_summarized)
    explainer = shap.KernelExplainer(cls.predict,X_summary)
    shap_values = explainer.shap_values(X)

    #explainer = shap.Explainer(f, med)
    #shap_values = explainer(X)

    return explainer, shap_values