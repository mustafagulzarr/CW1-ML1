import numpy as np


class Classifier:
    def __init__(self):
        # Initialise the input data based on each feature
        self.wallData = []
        self.foodData = []
        self.ghostData = []
        self.ghostInFrontData = []
        self.restData = []

        # Initialise the weights of the input based on each feature
        self.wallWeights = []
        self.foodWeights = []
        self.ghostWeights = []
        self.ghostInFrontWeights = []
        self.restWeights = []

        # Initialise the biases of the input based on each feature
        self.wallBias = 0
        self.foodBias = 0
        self.ghostBias = 0
        self.ghostInFrontBias = 0
        self.restBias = 0

        # Initialise the target data and number of classes (corresponds to each move: North, South, East, West)
        self.target = []
        self.num_classes = 0

    def reset(self):
        # Reset the input data based on each feature
        self.wallData = []
        self.foodData = []
        self.ghostData = []
        self.ghostInFrontData = []
        self.restData = []

        # Reset the weights of the input based on each feature
        self.wallWeights = []
        self.foodWeights = []
        self.ghostWeights = []
        self.ghostInFrontWeights = []
        self.restWeights = []

        # Reset the biases of the input based on each feature
        self.wallBias = 0
        self.foodBias = 0
        self.ghostBias = 0
        self.ghostInFrontBias = 0
        self.restBias = 0

        # Initialise the target data and number of classes (corresponds to each move: North, South, East, West)
        self.target = []
        self.num_classes = 0

    def fit(self, data, target):

        # Split input data into features
        self.wallData = self.splittingDataPerFeature(data, 0, 4)
        self.foodData = self.splittingDataPerFeature(data, 4, 8)
        self.ghostData = self.splittingDataPerFeature(data, 8, 16)
        self.ghostInFrontData = self.splittingDataPerFeature(data, 16, 16)
        self.restData = self.splittingDataPerFeature(data, 16, len(data))

        # Initialise target and number of classes
        self.target = np.array(target)
        self.num_classes = len(np.unique(target))

        # Create the weights and biases for each feature within the input data
        self.wallWeights, self.wallBias = self.weightsBiasPerFeature(self.wallData, self.target)
        self.foodWeights, self.foodBias = self.weightsBiasPerFeature(self.foodData, self.target)
        self.ghostWeights, self.ghostBias = self.weightsBiasPerFeature(self.ghostData, self.target)
        self.ghostInFrontWeights, self.ghostInFrontBias = self.weightsBiasPerFeature(self.ghostInFrontData, self.target)
        self.restWeights, self.restBias = self.weightsBiasPerFeature(self.restData, self.target)

    def predict(self, data, legal=None):

        # separate the input data into each feature
        wallData = data[0:4]
        foodData = data[4:8]
        ghostData = data[8:16]
        ghostInFrontData = data[16:16]
        restData = data[16:]

        # Create an empty scores array and appending the scores of each feature
        scores = []

        scoresWall = self.predictPerFeature(wallData, self.wallWeights, self.wallBias)
        scores.append(scoresWall)

        scoresFood = self.predictPerFeature(foodData, self.foodWeights, self.foodBias)
        scores.append(scoresFood)

        scoresGhost = self.predictPerFeature(ghostData, self.ghostWeights, self.ghostBias)
        scores.append(scoresGhost)

        scoresGhostInFront = self.predictPerFeature(ghostInFrontData, self.ghostInFrontWeights, self.ghostInFrontBias)
        scores.append(scoresGhostInFront)

        scoresRest = self.predictPerFeature(restData, self.restWeights, self.restBias)
        scores.append(scoresRest)

        # Return the index of the highest score among all features
        return np.argmax(scores)

    # This function splits the input data into different features
    @staticmethod
    def splittingDataPerFeature(data, start, end):

        # Initialise an empty input
        dataFeature = []

        # Splitting the data depending on start and finish
        for i in data:
            dataFeature.append(i[start:end])

        # Return the split data
        return np.array(dataFeature)

    # This function creates weights and biases for each feature based on input and target data
    def weightsBiasPerFeature(self, data, target):

        # Initialise an empty weights and bias
        weights = []
        bias = 0

        # Create the weights and bias of each feature based on SVM
        for i in range(self.num_classes):
            weights, bias = self.trainSVM(data, target)

        return weights, bias

    # This function predicts scores for each feature based on input data
    def predictPerFeature(self, dataFeature, weightsFeature, biasFeature):
        # Checks if the weights of each feature is null
        if len(weightsFeature) == 0:
            return 0

        # Compute the scores for each feature
        scoresPerFeature = []
        for i in range(self.num_classes):
            scorePerClass = np.dot(dataFeature, weightsFeature[i]) + biasFeature
            scoresPerFeature.append(scorePerClass)

        # Return the index of the highest score within each feature
        return np.argmax(scoresPerFeature)

    # This function fits the input data based on SVM
    def trainSVM(self, data, target):

        # get the number of samples and number of features from data
        numberSamples, numberFeatures = data.shape

        # initialize weights and bias
        weights = np.zeros(numberFeatures)
        bias = 0

        # learning rate
        LEARNING_RATE = 1

        # number of iterations
        NUMBER_OF_ITERATIONS = 100

        # gradient descent training
        # partially taken from https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/
        for i in range(NUMBER_OF_ITERATIONS):
            for j in range(numberSamples):
                if target[j] * (np.dot(weights, data[j]) + bias) <= 1:
                    weights += LEARNING_RATE * (target[j] * data[j] - 2 * weights / numberSamples)
                    bias += LEARNING_RATE * target[j]
                else:
                    weights += LEARNING_RATE * (- 2 * weights / numberSamples)
                    bias += LEARNING_RATE

        return weights, bias