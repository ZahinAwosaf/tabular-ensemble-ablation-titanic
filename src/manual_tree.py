from collections import Counter
import numpy as np

def gini(y):
    classes, counts = np.unique(y, return_counts = True)
    p = counts / counts.sum()
    
    return 1 - np.sum(p ** 2)


def entropy(y):
    classes, counts = np.unique(y, return_counts = True)
    p = counts / counts.sum()
    p = p[p > 0]
    
    return -np.sum(p * np.log2(p))


def information_gain(parent, left, right, criterion = "gini"):

    if criterion == "gini":
        impurity = gini
    elif criterion == "entropy":
        impurity = entropy
    else:
        raise ValueError("criterion must be 'gini' or 'entropy'")

    parent_impurity = impurity(parent)

    n = len(parent)
    n_left = len(left)
    n_right = len(right)
    weighted_child_impurity = ((n_left / n) * impurity(left) + (n_right / n) * impurity(right))

    return parent_impurity - weighted_child_impurity


class Node:

    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        

class DecisionTree:

    def __init__(self, max_depth = 3, min_samples_split = 2, criterion = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth = 0)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _best_split(self, X, y):

        n_samples, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):

            values = np.unique(X[:, feature])

            if len(values) < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:

                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = information_gain(y, y[left_mask], y[right_mask], self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):

        n_samples = len(y)
        n_classes = len(np.unique(y))

        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._most_common_label(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1,)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1,)

        return Node(feature = feature, threshold = threshold, left = left_child, right = right_child)

    def _traverse(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)

        return self._traverse(x, node.right)

    
def print_tree(node, depth=0):

    indent = "│   " * depth

    if node.value is not None:
        print(indent + f"└── Leaf: Class {node.value}")
        return

    print(indent + f"├── X[{node.feature}] <= {node.threshold:.3f}")

    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)