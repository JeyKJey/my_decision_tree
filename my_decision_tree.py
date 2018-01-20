from tree_grower import TreeGrower
from sklearn.base import BaseEstimator, ClassifierMixin
#import pandas as pd
import numpy as np

class MyDecisionTree(BaseEstimator, ClassifierMixin):
    """This module implements Desicion tree classifier."""

    def __init__(self, min_samples_split = 2, criterion = 'entropy'):
        if min_samples_split < 2:
            raise ValueError("min_samples_split should be more than 1")
        if criterion != 'entropy':
            ##TODO gini criterion
            raise ValueError("criterion should be entropy")

        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def fit(self, data, target):
        tree = TreeGrower(self.min_samples_split,
                          self.criterion)
        self.tree_ = tree.grow(data, target, data.columns.tolist())
        return self

    def predict(self, data):
        try:
            getattr(self, "tree_")
            predicts_proba = self.predict_proba(data)
            predicts = self._classify_from_probs(predicts_proba)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return predicts

    def predict_proba(self, data):
        try:
            getattr(self, "tree_")
            predicts = [self.tree_.traverse(row) for name, row in data.iterrows()]

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return predicts

    @staticmethod
    def _classify_from_probs(predicts_proba):
        def find_majority(dict_probs):
            #if there is no majority class, pick the first from the sorted
            max_val = max(dict_probs.values())
            max_keys = [key for key in dict_probs.keys()
                               if dict_probs[key] == max_val]
            return sorted(max_keys)[0]

        predicts = [find_majority(dict_probs) for dict_probs in predicts_proba]
        return predicts
