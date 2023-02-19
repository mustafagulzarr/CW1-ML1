# classifier.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#PREVIOUS CODE FOR CLASSIFIER
# class Classifier:
#     def _init_(self):
#         self.clf = None
#
#     def reset(self):
#         self.clf = None
#
#     def fit(self, data, target):
#         clf = RandomForestClassifier(n_estimators=100)
#         clf.fit(data, target)
#         self.clf = clf
#         # pass
#
#     def predict(self, data, legal=None):
#         clf = self.clf
#         pred = clf.predict([data])[0]
#         print(pred)
#         return pred

#CURRENT CODE FOR CLASSIFIER
class Classifier:
    def _init_(self):
        self.map1 = {}
        self.map2 = {}
        self.map3 = {}
        self.map4 = {}

    def reset(self):
        pass

    def fit(self, data, target):
        my_map1 = {}
        my_map2 = {}
        my_map3 = {}
        my_map4 = {}
        for i in range(len(data)):my_map1.setdefault((data[i][0],data[i][1],data[i][2],data[i][3]), []).append(target[i])
            my_map2.setdefault((data[i][4],data[i][5],data[i][6],data[i][7]), []).append(target[i])
            my_map3.setdefault((data[i][8],data[i][9],data[i][10],data[i][11],data[i][12],data[i][13],data[i][14],data[i][15]), []).append(target[i])
            my_map4.setdefault((data[i][16],data[i][17],data[i][18],data[i][19],data[i][20],data[i][21],data[i][22],data[i][23],data[i][24]), []).append(target[i])
        self.map1 = my_map1
        self.map2 = my_map2
        self.map3 = my_map3
        self.map4 = my_map4

    def predict(self, data, legal=None):
        my_map1 = self.map1
        my_map2 = self.map2
        my_map3 = self.map3
        my_map4 = self.map4

        # Predicting corresponding target given the location of walls
        s1 = my_map1.get((data[0],data[1],data[2],data[3]))
        counts1 = Counter(s1)

        # Predicting corresponding target given the location of food
        s2 = my_map2.get((data[4],data[5],data[6],data[7]))
        counts2 = Counter(s2)

        # Predicting corresponding target given the location of ghosts
        s3 = my_map3.get((data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15]))
        counts3 = Counter(s3)

        # Predicting corresponding target given the location of rest
        s4 = my_map4.get((data[16],data[17],data[18],data[19],data[20],data[21],data[22],data[23],data[24]))
        counts4 = Counter(s4)


        counterForDirection0 = 0
        counterForDirection1 = 0
        counterForDirection2 = 0
        counterForDirection3 = 0
        for key, value in counts1.items():
            if key == 0:
                counterForDirection0 += value
            if key == 1:
                counterForDirection1 += value
            elif key == 2:
                counterForDirection2 += value
            elif key == 3:
                counterForDirection3 += value

        for key, value in counts2.items():
            if key == 0:
                counterForDirection0 += value
            if key == 1:
                counterForDirection1 += value
            elif key == 2:
                counterForDirection2 += value
            elif key == 3:
                counterForDirection3 += value

        for key, value in counts3.items():
            if key == 0:
                counterForDirection0 += value
            if key == 1:
                counterForDirection1 += value
            elif key == 2:
                counterForDirection2 += value
            elif key == 3:
                counterForDirection3 += value

        for key, value in counts4.items():
            if key == 0:
                counterForDirection0 += value
            if key == 1:
                counterForDirection1 += value
            elif key == 2:
                counterForDirection2 += value
            elif key == 3:
                counterForDirection3 += value

        total = counterForDirection0+counterForDirection1+counterForDirection2+counterForDirection3

        probability_for_direction_0 = counterForDirection0/total
        probability_for_direction_1 = counterForDirection1/total
        probability_for_direction_2 = counterForDirection2/total
        probability_for_direction_3 = counterForDirection3/total
        listOfAllProbabilities = [counterForDirection0,counterForDirection1,counterForDirection2,counterForDirection3]

        pred= (listOfAllProbabilities.index(max(counterForDirection0,counterForDirection1,counterForDirection2,counterForDirection3)))

        return pred


# CHATBOT GPT CODE FOR CLASSIFIER

# class Classifier:
#     def __init__(self):
#         self.clf = None
#
#     def reset(self):
#         self.clf = None
#
#     def fit(self, data, target):
#         decision_tree = {}
#         # Get the number of features in the input data
#         num_features = len(data[0])
#
#         # Create a list of feature indices
#         feature_indices = list(range(num_features))
#
#         # Recursively build the decision tree
#         self.build_tree(np.array(data), np.array(target), feature_indices, decision_tree)
#
#         return decision_tree
#
#
#     def build_tree(self, X, y, feature_indices, decision_tree):
#         # Check if all the labels are the same
#         if len(set(y)) == 1:
#             decision_tree['label'] = y[0]
#             return
#
#         # Check if we have run out of features to split on
#         if len(feature_indices) == 0:
#             decision_tree['label'] = majority_vote(y)
#             return
#
#         # Find the best feature to split on
#         best_feature, best_threshold = self.find_best_split(X, y, feature_indices)
#
#         # Add the decision node to the tree
#         decision_tree['feature'] = best_feature
#         decision_tree['threshold'] = best_threshold
#
#         # Split the data based on the best feature and threshold
#         left_indices = X[:, best_feature] <= best_threshold
#         right_indices = X[:, best_feature] > best_threshold
#         left_X, left_y = X[left_indices], y[left_indices]
#         right_X, right_y = X[right_indices], y[right_indices]
#
#         # Recursively build the left and right subtrees
#         decision_tree['left'] = {}
#         self.build_tree(left_X, left_y, feature_indices, decision_tree['left'])
#         decision_tree['right'] = {}
#         self.build_tree(right_X, right_y, feature_indices, decision_tree['right'])
#
#     def find_best_split(self,X, y, feature_indices):
#         best_gain = -1
#         best_feature = None
#         best_threshold = None
#
#         # Loop over each feature and threshold
#         for feature in feature_indices:
#             for threshold in self.get_thresholds(X[:, feature]):
#                 left_indices = X[:, feature] <= threshold
#                 right_indices = X[:, feature] > threshold
#
#                 # print("lefty: ", left_indices)
#                 # print("righty: ", right_indices)
#                 # print("y: ", y)
#
#                 left_y = y[left_indices]
#                 right_y =  y[right_indices]
#
#                 # Calculate the information gain
#                 gain = self.information_gain(y, left_y, right_y)
#
#                 # Update the best feature and threshold if this split is better
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feature = feature
#                     best_threshold = threshold
#
#         return best_feature, best_threshold
#
#     def get_thresholds(self, feature_values):
#     # Sort the feature values
#         sorted_values = np.sort(feature_values)
#
#         # Get the midpoints between adjacent values
#         thresholds = (sorted_values[1:] + sorted_values[:-1]) / 2
#
#         return thresholds
#
#     def information_gain(self, parent_y, left_y, right_y):
#         # Calculate the entropy of the parent node
#         parent_entropy = self.entropy(parent_y)
#
#         # Calculate the entropy of the left and right child nodes
#         left_entropy = self.entropy(left_y)
#         right_entropy = self.entropy(right_y)
#
#         # Calculate the weighted average of the child entropies
#         num_left = len(left_y)
#         num_right = len(right_y)
#         total = num_left + num_right
#         left_weight = num_left / total
#         right_weight = num_right / total
#         child_entropy = left_weight * left_entropy + right_weight * right_entropy
#
#         # Calculate the information gain
#         gain = parent_entropy - child_entropy
#
#         return gain
#
#
#     def entropy(self, y):
#         # Calculate the proportion of each label in the input
#         counts = np.bincount(y)
#         proportions = counts / len(y)
#
#         # Calculate the entropy
#         entropy = 0
#         for p in proportions:
#             if p > 0:
#                 entropy -= p * np.log2(p)
#
#         return entropy
#
#
#     def predict(self, data, legal=None):
#         return 0
