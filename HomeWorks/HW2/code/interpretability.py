"""LIME and SHAP helpers for tabular models (uses model.predict_proba-like API).

- lime_explain: returns LIME explanation for a single sample
- shap_explain: returns SHAP values (KernelExplainer) for a few samples
"""
from typing import Any, List

import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
import shap


def lime_explain(model_predict_fn, X_train: np.ndarray, sample: np.ndarray, feature_names: List[str]):
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["neg", "pos"], mode="classification"
    )
    # Lime expects a 1D or 2D array for the instance
    exp = explainer.explain_instance(sample, model_predict_fn, num_features=len(feature_names))
    return exp


def shap_explain(model_predict_fn, X_background: np.ndarray, X_eval: np.ndarray):
    # KernelExplainer expects a function returning probability for each class;
    # our model_predict_fn should return prob of positive class
    explainer = shap.KernelExplainer(model_predict_fn, X_background[:100])
    vals = explainer.shap_values(X_eval, nsamples=200)
    return vals
