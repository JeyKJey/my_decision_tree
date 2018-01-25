"""This module creates and expands a tree """

import numpy as np
from criterion import calc_criterion, calc_probs

class TreeGrower():
    """
    Expand the tree with the criterions.

    :min_samples_split: Number of minimum samples to expkand the node.
    :criterion: Criterion for evaluation (gini, entropy)
    """

    def __init__(self,
                 min_samples_split,
                 criterion):
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def grow(self, data, target, features):
        """
        Return tree structure in form of a dictionary.

        Parameters
        ----------
        data : array_like matrix, shape = [n_samples, n_features]
        target : array, shape = [n_samples]
        features : array, shape = [n_features]

        Returns
        -------

        """
        self.tree = self.split_node(data, target, features)
        return self

    def split_node(self, data, target, features):
        """
        Return the node description.
        For inner node return:
            feature on which the node was splited,
            unique values of the feature,
            children nodes

        Parameters
        ----------
        data : array_like matrix, shape = [n_samples, n_features]
        target : array, shape = [n_samples]
        features : array, shape = [n_features]

        Returns
        -------

        """

        # if node is pure or min_samples_split is achieved
        if len(data) <= self.min_samples_split or len(np.unique(target)) == 1 or not features:
            probs = calc_probs(target)
            return {'Node type': 'Leaf', 'prob_value': probs}

        feats_entropy = [calc_criterion(data, feat, target, self.criterion)
                         for feat in features]

        feat_split = features[np.argmin(feats_entropy)]
        features = [feat for feat in features if feat != feat_split]

        feat_vals = np.unique(data[feat_split])
        nodes = []
        for feat_val in feat_vals:
            cond = data[feat_split] == feat_val
            x_loc = data[cond]
            y_loc = target[cond]
            nodes.append(self.split_node(x_loc, y_loc, features))

        return {'Node type': 'Innernode',
                'feature': feat_split,
                'feat_vals': feat_vals,
                'nodes': nodes
               }

    def traverse(self, row):
        """
        Return probabilities of a sample belogning to classes

        Parameters
        ----------
        row : array, shape = [n_features]

        Returns
        -------
        probability_values : probabilities of a sample belogning to classes

        """

        return _inner_traverse(self.tree, row)


def _inner_traverse(tree, row):
    """Unitility function to recursiverly traverse a tree"""

    if tree['Node type'] == 'Leaf':
        return tree['prob_value']
    feat_val = row[tree['feature']]
    index_feat_val = tree['feat_vals'] == feat_val
    tree_nodes = tree['nodes']
    sub_tree = tree_nodes[np.where(index_feat_val)[0][0]]
    return _inner_traverse(sub_tree, row)
