import numpy as np
from collections import Counter

def calculate_criterion(data, feature, target, criterion):
    if criterion == 'entropy':
        return calculate_entropy(data, feature, target)

def calculate_entropy(data, feature, target):
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

def count_probabilities(target):
    count_values = Counter(target)
    prob_values = {}
    for key in count_values:
        prob_values[key] = count_values[key]/len(target)
    return prob_values
