# classifier.py
# from collections import Counter
#
# #PREVIOUS CODE FOR CLASSIFIER
# # class Classifier:
# #     def _init_(self):
# #         self.clf = None
# #
# #     def reset(self):
# #         self.clf = None
# #
# #     def fit(self, data, target):
# #         clf = RandomForestClassifier(n_estimators=100)
# #         clf.fit(data, target)
# #         self.clf = clf
# #         # pass
# #
# #     def predict(self, data, legal=None):
# #         clf = self.clf
# #         pred = clf.predict([data])[0]
# #         print(pred)
# #         return pred
#
# #CURRENT CODE FOR CLASSIFIER
# class Classifier:
#     def _init_(self):
#         self.map1 = {}
#         self.map2 = {}
#         self.map3 = {}
#         self.map4 = {}
#
#     def reset(self):
#         pass
#
#     def fit(self, data, target):
#
#         for i in range(len(data)):
#             self.map1.setdefault((data[i][0],data[i][1],data[i][2],data[i][3]), []).append(target[i])
#             self.map2.setdefault((data[i][4],data[i][5],data[i][6],data[i][7]), []).append(target[i])
#             self.map3.setdefault((data[i][8],data[i][9],data[i][10],data[i][11],data[i][12],data[i][13],data[i][14],data[i][15]), []).append(target[i])
#             self.map4.setdefault((data[i][16],data[i][17],data[i][18],data[i][19],data[i][20],data[i][21],data[i][22],data[i][23],data[i][24]), []).append(target[i])
#
#     def predict(self, data, legal=None):
#
#         # Predicting corresponding target given the location of walls
#         s1 = self.map1.get((data[0],data[1],data[2],data[3]))
#         counts1 = Counter(s1)
#
#         # Predicting corresponding target given the location of food
#         s2 = self.map2.get((data[4],data[5],data[6],data[7]))
#         counts2 = Counter(s2)
#
#         # Predicting corresponding target given the location of ghosts
#         s3 = self.map3.get((data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15]))
#         counts3 = Counter(s3)
#
#         # Predicting corresponding target given the location of rest
#         s4 = self.map4.get((data[16],data[17],data[18],data[19],data[20],data[21],data[22],data[23],data[24]))
#         counts4 = Counter(s4)
#
#         counterForDirection0 = 0
#         counterForDirection1 = 0
#         counterForDirection2 = 0
#         counterForDirection3 = 0
#
#         for key, value in counts1.items():
#             if key == 0:
#                 counterForDirection0 += value
#             if key == 1:
#                 counterForDirection1 += value
#             elif key == 2:
#                 counterForDirection2 += value
#             elif key == 3:
#                 counterForDirection3 += value
#
#         for key, value in counts2.items():
#             if key == 0:
#                 counterForDirection0 += value
#             if key == 1:
#                 counterForDirection1 += value
#             elif key == 2:
#                 counterForDirection2 += value
#             elif key == 3:
#                 counterForDirection3 += value
#
#         for key, value in counts3.items():
#             if key == 0:
#                 counterForDirection0 += value
#             if key == 1:
#                 counterForDirection1 += value
#             elif key == 2:
#                 counterForDirection2 += value
#             elif key == 3:
#                 counterForDirection3 += value
#
#         for key, value in counts4.items():
#             if key == 0:
#                 counterForDirection0 += value
#             if key == 1:
#                 counterForDirection1 += value
#             elif key == 2:
#                 counterForDirection2 += value
#             elif key == 3:
#                 counterForDirection3 += value
#
#         total = counterForDirection0+counterForDirection1+counterForDirection2+counterForDirection3
#
#         probability_for_direction_0 = counterForDirection0/total
#         probability_for_direction_1 = counterForDirection1/total
#         probability_for_direction_2 = counterForDirection2/total
#         probability_for_direction_3 = counterForDirection3/total
#         listOfAllProbabilities = [counterForDirection0,counterForDirection1,counterForDirection2,counterForDirection3]
#
#         pred = (listOfAllProbabilities.index(max(counterForDirection0,counterForDirection1,counterForDirection2,counterForDirection3)))
#
#         return pred


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

# # first attempt
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.kernel = None
#         self.C = 1.0
#
#     def reset(self):
#         self.alpha = None
#         self.b = None
#
#     def fit(self, data, target):
#         n = len(data)
#         K = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 K[i,j] = self.kernel
#
#         P = target.reshape(-1,1) @ target.reshape(1,-1) * K
#         q = -np.ones(n)
#         G = np.vstack((-np.eye(n), np.eye(n)))
#         h = np.hstack((np.zeros(n), self.C * np.ones(n)))
#         A = target.reshape(1,-1)
#         b = np.array([0.0])
#
#         from cvxopt import matrix, solvers
#         solvers.options['show_progress'] = False
#         sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
#         self.alpha = np.array(sol['x']).reshape(-1)
#         sv = self.alpha > 1e-5
#         self.support_vectors = data[sv]
#         self.support_targets = target[sv]
#         self.alpha = self.alpha[sv]
#
#         s = np.zeros(n)
#         for i in range(len(self.alpha)):
#             s += self.alpha[i] * self.support_targets[i] * K[sv[i], :]
#         self.b = np.mean(self.support_targets - s)
#
#     def predict(self, data, legal=None):
#         n = len(data)
#         K = np.zeros((n, len(self.support_vectors)))
#         for i in range(n):
#             for j in range(len(self.support_vectors)):
#                 K[i,j] = self.kernel(data[i], self.support_vectors[j])
#
#         prediction = np.sign(np.sum(self.alpha * self.support_targets * K, axis=1) + self.b)
#
#         if legal is not None:
#             legal_actions = np.array(legal)
#             illegal_actions = np.array([0, 1, 2, 3]) - legal_actions
#             prediction[np.isin(prediction, illegal_actions)] = 0
#
#         return prediction


# second attempt
# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import rbf_kernel
#
# class Classifier:
#     def __init__(self):
#         self.clf = make_pipeline(StandardScaler(), SVC(kernel='precomputed'))
#
#     def reset(self):
#         self.clf = make_pipeline(StandardScaler(), SVC(kernel='precomputed'))
#
#     def fit(self, data, target):
#         # Compute RBF kernel matrix between the training data points
#         K = rbf_kernel(data, gamma=0.1)
#
#         # Fit the classifier using the kernel matrix and target values
#         self.clf.fit(K, target)
#
#     def predict(self, data, legal=None):
#         if legal is None:
#             return self.clf.predict(data.reshape(1, -1))[0]
#         else:
#             # Compute RBF kernel between the training data and the test data
#             K = rbf_kernel(data.reshape(1, -1), self.clf.named_steps['standardscaler'].transform(self.clf.named_steps['svc'].support_vectors_), gamma=0.1)
#             # Compute the predicted target value
#             pred = self.clf.predict(K)
#             # If the predicted action is illegal, choose a random legal action
#             if self.convertNumberToMove(pred[0]) not in legal:
#                 return self.convertMoveToNumber(random.choice(legal))
#             else:
#                 return pred[0]
#
#     def convertNumberToMove(self, number):
#         if number == 0:
#             return Directions.NORTH
#         elif number == 1:
#             return Directions.EAST
#         elif number == 2:
#             return Directions.SOUTH
#         elif number == 3:
#             return Directions.WEST
#
#     def convertMoveToNumber(self, move):
#         if move == Directions.NORTH:
#             return 0
#         elif move == Directions.EAST:
#             return 1
#         elif move == Directions.SOUTH:
#             return 2
#         elif move == Directions.WEST:
#             return 3

#
# #third attempt
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.gamma = 0.1
#         self.num_classes = 4
#         print(self.num_classes)
#         self.centers = None
#         self.weights = None
#
#     def reset(self):
#         self.centers = None
#         self.weights = None
#
#     def _rbf_kernel(self, X1, X2):
#         # Compute pairwise distances between the rows of X1 and X2
#         dists = np.sum(X1**2) + np.sum(X2**2) - 2 * X1.dot(X2.T)
#         #axis=1
#         #.reshape(-1, 1)
#
#         # Apply the RBF kernel function
#         return np.exp(-self.gamma * dists)
#
#     def fit(self, data, target):
#         # Convert the target array into a one-hot encoding matrix
#         print(self.num_classes)
#         targets = np.zeros((len(target), self.num_classes))
#         for i, target_val in enumerate(target):
#             targets[i, target_val] = 1
#
#         # Randomly select centers from the data points
#         num_centers = min(1000, len(data))
#         center_indices = np.random.choice(len(data), num_centers, replace=False)
#         print(center_indices)
#         dataArray = np.array(data)
#         self.centers = dataArray[center_indices]
#
#         # Compute the kernel matrix between the data and the centers
#         X = self._rbf_kernel(dataArray, self.centers)
#
#         # Add a column of ones for the bias term
#         #X = np.column_stack([np.ones((X.shape[0], 1)), X])
#
#         # Compute the weights using the Moore-Penrose pseudoinverse
#         self.weights = np.linalg.pinv(X).dot(targets)
#
#     def predict(self, data, legal=None):
#         dataArray = np.array(data)
#         # Compute the kernel matrix between the data and the centers
#         X = self._rbf_kernel(dataArray, self.centers)
#
#         # Add a column of ones for the bias term
#         #print(np.ones((X.shape[0])))
#         ##print(X)
#         X = np.column_stack([np.ones((X.shape[0], 1)), X])
#         #print(X.shape)
#         #X = np.transpose
#
#         # Compute the predicted class probabilities
#         print(self.weights[:, :2].shape)
#         print(X.shape)
#         y = X.dot(self.weights[:, :2].transpose())
#
#         # Convert the probabilities to class labels
#         return np.argmax(y, axis=0)


# # classifier.py
# # Your Name/21-Feb-2023
# #
# # Use the skeleton below for the classifier and insert your code here.
#
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.kernel = None
#         self.alpha = None
#         self.data = None
#         self.target = None
#
#     def reset(self):
#         self.alpha = None
#         self.data = None
#         self.target = None
#
#     def fit(self, data, target):
#         self.data = np.array(data)
#         self.target = np.array(target)
#         n = len(data)
#
#         # compute kernel matrix
#         K = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 K[i, j] = self.kernel(data[i], data[j])
#
#         # solve for alpha using quadratic programming
#         P = np.outer(target, target) * K
#         q = -np.ones(n)
#         G = np.diag(-np.ones(n))
#         h = np.zeros(n)
#         A = target.reshape((1, n))
#         b = np.array([0.0])
#         from cvxopt import matrix, solvers
#         solvers.options['show_progress'] = False
#         alpha = np.array(solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))['x']).flatten()
#
#         self.alpha = alpha
#
#     def predict(self, data, legal=None):
#         if self.alpha is None:
#             return 0
#
#         data = np.array(data)
#         n = len(data)
#
#         # compute kernel vector
#         k = np.zeros(n)
#         for i in range(len(self.data)):
#             k += self.alpha[i] * self.kernel(self.data[i], data)
#
#         return np.sign(k.sum())
#
# # working attempt 1
# import numpy as np
# from collections import Counter
#
# class Classifier:
#     def __init__(self):
#         self.k = 3
#         self.data = None
#         self.target = None
#
#     def reset(self):
#         self.data = None
#         self.target = None
#
#     def fit(self, data, target):
#         self.data = np.array(data)
#         self.target = np.array(target)
#
#     def predict(self, data, legal=None):
#         predictions = []
#         for x in data:
#             # Calculate distances to all data points
#             distances = np.sum((self.data - x)**2, axis=1)
#             # Get the indices of the k nearest data points
#             indices = np.argsort(distances)[:self.k]
#             # Get the target values of the k nearest data points
#             targets = self.target[indices]
#             # Count the occurrences of each target value
#             counts = Counter(targets)
#             # Get the most common target value (break ties randomly)
#             most_common = counts.most_common()
#             prediction = np.random.choice([mc[0] for mc in most_common if mc[1] == most_common[0][1]])
#             predictions.append(prediction)
#         return predictions
#
# # #working attempt 2
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.weights = None
#
#     def reset(self):
#         self.weights = None
#
#     def fit(self, data, target):
#         num_features = len(data[0])
#         num_classes = len(np.unique(target))
#         num_samples = len(data)
#
#         # Initialize weights randomly
#         self.weights = np.random.randn(num_features, num_classes)
#
#         # One-hot encode the target labels
#         target_onehot = np.zeros((num_samples, num_classes))
#         target_onehot[np.arange(num_samples), target] = 1
#
#         # Train the weights using perceptron learning algorithm
#         learning_rate = 0.1
#         num_iterations = 100
#         for i in range(num_iterations):
#             for j in range(num_samples):
#                 y_pred = np.dot(data[j], self.weights)
#                 error = target_onehot[j] - y_pred
#                 delta = learning_rate * np.outer(data[j], error)
#                 self.weights += delta
#
#     def predict(self, data, legal=None):
#         if self.weights is None:
#             return 0  # Default action if not yet trained
#
#         # Calculate scores for each possible action
#         scores = np.dot(data, self.weights)
#
#         # Find the action with the highest score
#         if legal is None:
#             return np.argmax(scores)
#         else:
#             legal_mask = np.zeros_like(scores, dtype=bool)
#             #legal_mask[legal] = True
#             scores[~legal_mask] = -np.inf
#             print(np.argmax(scores))
#             return np.argmax(scores)
#
# not working
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.learning_rate = 0.1
#         self.max_iterations = 1000
#         self.weights = None
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def fit(self, data, target):
#         # Add a column of ones to X to represent the bias term
#         self.data = np.c_[np.ones((data.shape[0], 1)), data]
#
#         # Initialize weights to zeros
#         self.weights = np.zeros(data.shape[1])
#
#         # Gradient descent
#         for i in range(self.max_iterations):
#             # Compute the predicted probabilities
#             y_pred = self.sigmoid(data @ self.weights)
#
#             # Compute the gradient of the cost function
#             gradient = X.T @ (y_pred - target)
#
#             # Update the weights
#             self.weights -= self.learning_rate * gradient
#
#     def predict(self, data, legal=None):
#         # Add a column of ones to X to represent the bias term
#         self.data = np.c_[np.ones((data.shape[0], 1)), data]
#
#         # Compute the predicted probabilities
#         y_pred = self.sigmoid(data @ self.weights)
#
#         # Convert probabilities to classes
#         y_pred_class = np.round(y_pred)
#
#         # If legal actions are provided, set illegal actions to 0 probability
#         if legal is not None:
#             legal_mask = np.zeros(4, dtype=bool)
#             #legal_mask[legal] = True
#             y_pred_class[~legal_mask] = 0
#
#         # Return the predicted classes
#         return y_pred_class.astype(int)

# import numpy as np
#
# class Classifier:
#     def __init__(self, learning_rate=0.1, max_iterations=1000):
#         self.learning_rate = learning_rate
#         self.max_iterations = max_iterations
#         self.weights = None
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def fit(self, X, y):
#         # Add a column of ones to X to represent the bias term
#         X = np.c_[np.ones((X.shape[0], 1)), X]
#
#         # Initialize weights to zeros
#         self.weights = np.zeros(X.shape[1])
#
#         # Gradient descent
#         for i in range(self.max_iterations):
#             # Compute the predicted probabilities
#             y_pred = self.sigmoid(X @ self.weights)
#
#             # Compute the gradient of the cost function
#             gradient = X.T @ (y_pred - y)
#
#             # Update the weights
#             self.weights -= self.learning_rate * gradient
#
#     def predict(self, X, legal=None):
#         # Add a column of ones to X to represent the bias term
#         X = np.c_[np.ones((X.shape[0], 1)), X]
#
#         # Compute the predicted probabilities
#         y_pred = self.sigmoid(X @ self.weights)
#
#         # Convert probabilities to classes
#         y_pred_class = np.round(y_pred)
#
#         # If legal actions are provided, set illegal actions to 0 probability
#         if legal is not None:
#             legal_mask = np.zeros(4, dtype=bool)
#             legal_mask[legal] = True
#             y_pred_class[~legal_mask] = 0
#
#         # Return the predicted classes
#         return y_pred_class.astype(int)
#
# # working perceptron implementation with 1 hidden layer and bayes
# import numpy as np
#
# class Classifier:
#     def __init__(self):
#         self.weights = None
#
#     def reset(self):
#         self.weights = None
#
#     def fit(self, data, target):
#         # Convert data and target to numpy arrays
#         learning_rate = 0.1
#         num_epochs = 100
#         data = np.array(data)
#         target = np.array(target)
#
#         # Add a bias term to the data
#         data = np.insert(data, 0, 1, axis=1)
#
#         # Initialize weights to zeros
#         num_features = data.shape[1]
#         self.weights = np.zeros(num_features)
#
#         # Train the classifier using stochastic gradient descent
#         for epoch in range(num_epochs):
#             for i in range(data.shape[0]):
#                 x = data[i]
#                 y = target[i]
#
#                 # Compute the predicted output
#                 z = np.dot(self.weights, x)
#                 y_pred = np.sign(z)
#
#                 # Update the weights if the prediction is incorrect
#                 if y != y_pred:
#                     self.weights += learning_rate * y * x
#
#     def predict(self, data, legal=None):
#         # Convert data to numpy array
#         data = np.array(data)
#
#         # Add a bias term to the data
#         data = np.insert(data, 0, 1)
#
#         # Compute the predicted outputs
#         z = np.dot(data, self.weights)
#         y_pred = np.sign(z)
#
#         # Convert the predicted outputs to actions
#         actions = np.where(y_pred == 1)[0]
#         if legal is not None:
#             actions = np.intersect1d(actions, legal)
#         if len(actions) > 0:
#             return actions[0]
#         else:
#             return np.random.choice(legal)

# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

# import numpy as np
#
#
# class Classifier:
#     def __init__(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def reset(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def fit(self, data, target):
#         data = np.array(data)
#         hidden_layers = [10]
#         self.n_layers = len(hidden_layers) + 1
#
#         # Initialize weights and biases randomly
#         input_size = data.shape[1]
#         output_size = len(np.unique(target))
#         layer_sizes = [input_size] + hidden_layers + [output_size]
#         for i in range(self.n_layers):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
#             self.biases.append(np.random.randn(layer_sizes[i + 1]))
#
#         # Train the network using stochastic gradient descent
#         learning_rate = 0.1
#         n_epochs = 1000
#         batch_size = 32
#         n_batches = int(np.ceil(data.shape[0] / batch_size))
#         for epoch in range(n_epochs):
#             for b in range(n_batches):
#                 # Get the batch of data and target values
#                 batch_data = data[b * batch_size:(b + 1) * batch_size, :]
#                 batch_target = target[b * batch_size:(b + 1) * batch_size]
#
#                 # Forward propagation
#                 a = batch_data
#                 for i in range(self.n_layers):
#                     z = np.dot(a, self.weights[i]) + self.biases[i]
#                     a = self.sigmoid(z)
#
#                 # Backward propagation
#                 delta = (a - self.one_hot_encode(batch_target)) * self.sigmoid_derivative(a)
#                 for i in reversed(range(self.n_layers)):
#                     grad_w = np.dot(a.T, delta)
#                     grad_b = np.sum(delta, axis=0)
#                     delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z)
#                     self.weights[i] -= learning_rate.T * grad_w
#                     self.biases[i] -= learning_rate * grad_b
#
#     def predict(self, data, legal=None):
#         # Perform feedforward to get predicted output
#         a = data
#         for i in range(self.n_layers):
#             z = np.dot(a, self.weights[i]) + self.biases[i]
#             a = self.sigmoid(z)
#         predicted_output = np.argmax(a, axis=1)
#         return predicted_output
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         return x * (1 - x)
#
#     def one_hot_encode(self, y):
#         n_classes = len(np.unique(y))
#         encoded = np.zeros((len(y), n_classes))
#         for i in range(len(y)):
#             encoded[i, y[i]] = 1
#         return encoded

#
# import numpy as np
#
#
# class Classifier:
#     def __init__(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def reset(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def fit(self, data, target):
#         hidden_layers = [10]
#         data = np.array(data)
#         self.n_layers = len(hidden_layers) + 1
#
#         # Initialize weights and biases randomly
#         input_size = data.shape[1]
#         output_size = len(np.unique(target))
#         layer_sizes = [input_size] + hidden_layers + [output_size]
#         for i in range(self.n_layers):
#             self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]))
#             self.biases.append(np.random.randn(layer_sizes[i + 1], 1))
#
#         # Train the network using stochastic gradient descent
#         learning_rate = 0.1
#         n_epochs = 1000
#         batch_size = 32
#         n_batches = int(np.ceil(data.shape[0] / batch_size))
#         for epoch in range(n_epochs):
#             for b in range(n_batches):
#                 # Get the batch of data and target values
#                 batch_data = data[b * batch_size:(b + 1) * batch_size, :]
#                 batch_target = target[b * batch_size:(b + 1) * batch_size]
#
#                 # Forward propagation
#                 a = batch_data.T
#                 z_list = []
#                 for i in range(self.n_layers):
#                     z = np.dot(a, self.weights[i]) + self.biases[i]
#                     z_list.append(z)
#                     a = self.sigmoid(z)
#
#                 # Backward propagation
#                 delta = (a - self.one_hot_encode(batch_target).T) * self.sigmoid_derivative(a)
#                 for i in reversed(range(self.n_layers)):
#                     grad_w = np.dot(a.T, delta)
#                     grad_b = np.sum(delta, axis=1, keepdims=True)
#                     delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z_list[i - 1]) if i > 0 else delta
#                     print(learning_rate * grad_w)
#                     print(self.weights[i])
#                     self.weights[i] -= (learning_rate * grad_w)
#                     self.biases[i] -= (learning_rate * grad_b)
#
#     # def fit(self, data, target, hidden_layers=[10]):
#     #
#     #     data = np.array(data)
#     #     self.n_layers = len(hidden_layers) + 1
#     #
#     #     # Initialize weights and biases randomly
#     #     input_size = data.shape[1]
#     #     output_size = len(np.unique(target))
#     #     layer_sizes = [input_size] + hidden_layers + [output_size]
#     #     for i in range(self.n_layers):
#     #         self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]))
#     #         self.biases.append(np.random.randn(layer_sizes[i + 1], 1))
#     #
#     #     # Train the network using stochastic gradient descent
#     #     learning_rate = 0.1
#     #     n_epochs = 1000
#     #     batch_size = 32
#     #     n_batches = int(np.ceil(data.shape[0] / batch_size))
#     #     for epoch in range(n_epochs):
#     #         for b in range(n_batches):
#     #             # Get the batch of data and target values
#     #             batch_data = data[b * batch_size:(b + 1) * batch_size, :]
#     #             batch_target = target[b * batch_size:(b + 1) * batch_size]
#     #
#     #             # Forward propagation
#     #             a = batch_data.T
#     #             z_list = []
#     #             for i in range(self.n_layers):
#     #                 z = np.dot(self.weights[i], a) + self.biases[i]
#     #                 z_list.append(z)
#     #                 a = self.sigmoid(z)
#     #
#     #             # Backward propagation
#     #             delta = (a - self.one_hot_encode(batch_target).T) * self.sigmoid_derivative(a)
#     #             for i in reversed(range(self.n_layers)):
#     #                 grad_w = np.dot(delta, a.T)
#     #                 grad_b = np.sum(delta, axis=1, keepdims=True)
#     #                 delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(
#     #                     z_list[i - 1]) if i > 0 else delta
#     #                 self.weights[i] -= learning_rate * grad_w
#     #                 self.biases[i] -= learning_rate * grad_b
#
#     def predict(self, data, legal=None):
#         # Perform feedforward to get predicted output
#         a = data
#         for i in range(self.n_layers):
#             z = np.dot(a, self.weights[i]) + self.biases[i]
#             a = self.sigmoid(z)
#         predicted_output = np.argmax(a, axis=1)
#         return predicted_output
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         return x * (1 - x)
#
#     def one_hot_encode(self, y):
#         n_classes = len(np.unique(y))
#         encoded = np.zeros((len(y), n_classes))
#         for i in range(len(y)):
#             encoded[i, y[i]] = 1
#         return encoded
#
# import numpy as np
#
#
# class Classifier:
#     def __init__(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def reset(self):
#         self.weights = []
#         self.biases = []
#         self.n_layers = 0
#
#     def fit(self, data, target):
#         hidden_layers = [10]
#         data = np.array(data)
#         self.n_layers = len(hidden_layers) + 1
#
#         # Initialize weights and biases randomly
#         input_size = data.shape[1]
#         output_size = len(np.unique(target))
#         layer_sizes = [input_size] + hidden_layers + [output_size]
#         for i in range(self.n_layers):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
#             self.biases.append(np.random.randn(layer_sizes[i + 1]))
#
#         # Train the network using stochastic gradient descent
#         learning_rate = 0.1
#         n_epochs = 1000
#         batch_size = 32
#         n_batches = int(np.ceil(data.shape[0] / batch_size))
#         for epoch in range(n_epochs):
#             for b in range(n_batches):
#                 # Get the batch of data and target values
#                 batch_data = data[b * batch_size:(b + 1) * batch_size, :]
#                 batch_target = target[b * batch_size:(b + 1) * batch_size]
#
#                 # Forward propagation
#                 a = batch_data
#                 z_list = []
#                 for i in range(self.n_layers):
#                     z = np.dot(self.weights[i], a) + self.biases[i]
#                     z_list.append(z)
#                     a = self.sigmoid(z)
#
#                 # Backward propagation
#                 delta = (a - self.one_hot_encode(batch_target)) * self.sigmoid_derivative(a)
#                 for i in reversed(range(self.n_layers)):
#                     grad_w = np.dot(a.T, delta)
#                     grad_b = np.sum(delta, axis=0)
#                     # delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z)
#                     delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(z_list[i - 1]) if i > 0 else delta
#                     self.weights[i] -= learning_rate * grad_w.T
#                     self.biases[i] -= learning_rate * grad_b
#
#     def predict(self, data, legal=None):
#         # Perform feedforward to get predicted output
#         a = data
#         for i in range(self.n_layers):
#             z = np.dot(a, self.weights[i]) + self.biases[i]
#             a = self.sigmoid(z)
#         predicted_output = np.argmax(a, axis=1)
#         return predicted_output
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         return x * (1 - x)
#
#     def one_hot_encode(self, y):
#         n_classes = len(np.unique(y))
#         encoded = np.zeros((len(y), n_classes))
#         for i in range(len(y)):
#             encoded[i, y[i]] = 1
#         return encoded
#
# import numpy as np
# from sklearn.svm import LinearSVC
#
# class Classifier:
#     def __init__(self):
#         self.svm = None
#
#     def reset(self):
#         self.svm = None
#
#     def fit(self, data, target):
#         # Convert data and target to numpy arrays
#         data = np.array(data)
#         target = np.array(target)
#
#         # Train a linear SVM model
#         self.svm = LinearSVC(max_iter=1000, tol=1e-3)
#         self.svm.fit(data, target)
#
#     def predict(self, data, legal=None):
#         # Convert data to numpy array
#         data = np.array(data)
#
#         # Predict the outputs using the trained SVM model
#         y_pred = self.svm.predict(data)
#
#         # Convert the predicted outputs to actions
#         actions = np.where(y_pred == 1)[0]
#         if legal is not None:
#             actions = np.intersect1d(actions, legal)
#         if len(actions) > 0:
#             return actions[0]
#         else:
#             return np.random.choice(legal)

import numpy as np

class Classifier:
    def __init__(self):
        self.wallData = []
        self.foodData = []
        self.ghostData = []
        self.ghostInFrontData = []
        self.restData = []

        self.target = []
        self.num_classes = 0

        self.wallWeights = []
        self.foodWeights = []
        self.ghostWeights = []
        self.ghostInFrontWeights = []
        self.restWeights = []

        self.wallBias = 0
        self.foodBias = 0
        self.ghostBias = 0
        self.ghostInFrontBias = 0
        self.restBias = 0

    def reset(self):
        self.wallData = []
        self.foodData = []
        self.ghostData = []
        self.ghostInFrontData = []
        self.restData = []

        self.target = []
        self.num_classes = 0

        self.wallWeights = []
        self.foodWeights = []
        self.ghostWeights = []
        self.ghostInFrontWeights = []
        self.restWeights = []

        self.wallBias = 0
        self.foodBias = 0
        self.ghostBias = 0
        self.ghostInFrontBias = 0
        self.restBias = 0

    def fit(self, data, target):

        print(data)
        self.wallData = self.splittingData(data, 0, 4)
        self.foodData = self.splittingData(data, 4, 8)
        self.ghostData = self.splittingData(data, 8, 16)
        self.ghostInFrontData = self.splittingData(data, 16, 16)
        self.restData = self.splittingData(data, 16, len(data))

        self.target = np.array(target)
        self.num_classes = len(np.unique(target))

        # one-vs-all multiclass SVM
        self.wallWeights, self.wallBias = self.weightsBiasPerClass(self.wallData, self.target)
        self.foodWeights, self.foodBias = self.weightsBiasPerClass(self.foodData, self.target)
        self.ghostWeights, self.ghostBias = self.weightsBiasPerClass(self.ghostData, self.target)
        self.ghostInFrontWeights, self.ghostInFrontBias = self.weightsBiasPerClass(self.ghostInFrontData, self.target)
        self.restWeights, self.restBias = self.weightsBiasPerClass(self.restData, self.target)

    def weightsBiasPerClass(self, data, target):
        weights = []
        bias = 0
        for i in range(self.num_classes):
            targets = np.where(target == i, 1, -1)
            weights, bias = self._svm_train(data, targets)
        return weights, bias

    @staticmethod
    def splittingData(data, start, end):
        dataCategory = []
        for i in data:
            dataCategory.append(i[start:end])
        return np.array(dataCategory)

    def predict(self, data, legal=None):

        wallData = data[0:4]
        foodData = data[4:8]
        ghostData = data[8:16]
        ghostInFrontData = data[16:16]
        restData = data[16:]

        scores = []
        scoresWall = self.predictPerClass(wallData, self.wallWeights, self.wallBias)
        scores.append(scoresWall)
        scoresFood = self.predictPerClass(foodData, self.foodWeights, self.foodBias)
        scores.append(scoresFood)
        scoresGhost = self.predictPerClass(ghostData, self.ghostWeights, self.ghostBias)
        scores.append(scoresGhost)
        scoresGhostInFront = self.predictPerClass(ghostInFrontData, self.ghostInFrontWeights, self.ghostInFrontBias)
        scores.append(scoresGhostInFront)
        scoresRest = self.predictPerClass(restData, self.wallWeights, self.wallBias)
        scores.append(scoresRest)

        return np.argmax(scores)

    def predictPerClass(self, dataCategory, weightsCategory, biasCategory):

        # Checks if the weights of each category is null
        if len(weightsCategory) == 0:
            return 0

        # compute scores for each class
        scoresPerClass = []
        for i in range(self.num_classes):
            scorePerClass = np.dot(dataCategory, weightsCategory[i]) + biasCategory
            scoresPerClass.append(scorePerClass)

        # return the highest score within each class
        return np.argmax(scoresPerClass)

    def _svm_train(self, data, targets):
        num_samples, num_features = data.shape

        # initialize weights and bias
        weights = np.zeros(num_features)
        bias = 0

        # learning rate
        learningRate = 1

        # number of iterations
        num_iters = 100

        # gradient descent training
        for i in range(num_iters):
            for j in range(num_samples):
                if targets[j] * (np.dot(weights, data[j]) + bias) <= 1:
                    weights += learningRate * (targets[j] * data[j] - 2 * weights / num_samples)
                    bias += learningRate * targets[j]
        return weights, bias
