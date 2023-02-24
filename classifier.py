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
