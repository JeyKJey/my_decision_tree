"""This module implements Criterions."""
##TODO: Implement it as a class

from collections import Counter
import numpy as np

def calc_criterion(data, feature, target, criterion):
    """
    Returns calculated criterion for a given feature.

    Parameters
    ----------
    data : array_like matrix, shape = [n_samples, n_features]
    target : array, shape = [n_samples]
    feature : string

    Returns
    -------
    calculated criterion
    """
    if criterion == 'entropy':
        return calc_entropy(data, feature, target)
    else:
        pass

def calc_entropy(data, feature, target):
    """
    Calculates entropy for a given feature.

    Parameters
    ----------
    data : array_like matrix, shape = [n_samples, n_features]
    target : array, shape = [n_samples]
    feature : string

    Returns
    -------
    entropy for a given feature
    """
    entropies = np.empty(shape=0)
    weights = np.empty(shape=0)
    for unique_value in np.unique(data[feature]):
        y_loc = target[data[feature] == unique_value]

        probs = np.array(list(Counter(y_loc).values()))/len(y_loc)
        entropy = - np.dot(probs, np.log2(probs)).sum()
        entropies = np.append(entropies, entropy)

        weight = len(y_loc)/len(target)
        weights = np.append(weights, weight)

    return np.dot(entropies, weights)

def calc_probs(target):
    """
    Calculates probabilites for each class withint the given data subset

    Parameters
    ----------
    target : array, shape = [n_samples]
        Input samples

    Returns
    -------
    prob_values : dict,
        The class probabilities of the input samples
    """
    count_values = Counter(target)
    prob_values = {}
    for key in count_values:
        prob_values[key] = count_values[key]/len(target)
    return prob_values
