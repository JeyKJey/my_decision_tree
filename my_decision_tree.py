"""This module implements Desicion tree classifier."""

from sklearn.base import BaseEstimator, ClassifierMixin
from tree_grower import TreeGrower

class MyDecisionTree(BaseEstimator, ClassifierMixin):
    """This class implements Desicion tree classifier."""

    def __init__(self, min_samples_split=2, criterion='entropy'):
        if min_samples_split < 2:
            raise ValueError("min_samples_split should be more than 1")
        if criterion != 'entropy':
            ##TODO gini criterion
            raise ValueError("criterion should be entropy")

        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, data, target):
        """Builds a decision tree classifier from the training set (data, target).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
        """
        tree = TreeGrower(self.min_samples_split,
                          self.criterion)
        self.tree = tree.grow(data, target, data.columns.tolist())
        return self

    def predict(self, data):
        """Predicts class for the samples from data.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]

        Returns
        -------
        predicts : array of shape = [n_samples]
            The predicted classes
        """
        try:
            getattr(self, "tree_")
            predicts_proba = self.predict_proba(data)
            predicts = _classify_from_probs(predicts_proba)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return predicts

    def predict_proba(self, data):
        """Predicts probabilities belogning to each class for the samples from data.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]

        Returns
        -------
        predicts_proba : array of shape = [n_samples]
        The predicted probabilities of belogning to each class
        """
        try:
            getattr(self, "tree_")
            predicts = [self.tree.traverse(row) for name, row in data.iterrows()]

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return predicts


def _classify_from_probs(predicts_proba):
    """Utility function to extract class from the probabilities"""
    def find_majority(dict_probs):
        """Finds the majority class"""
        # if there is no majority class, pick the first from the sorted
        max_val = max(dict_probs.values())
        max_keys = [key for key in dict_probs.keys()
                    if dict_probs[key] == max_val]
        return sorted(max_keys)[0]

    predicts = [find_majority(dict_probs) for dict_probs in predicts_proba]
    return predicts
