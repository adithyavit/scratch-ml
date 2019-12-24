
import numpy as np # linear algebra
import math
from scipy.io import loadmat

#load the data
data= loadmat(‘mnist_data.mat’) 

#Load trX, trY, tsX, tsY
trX = data['trX']
print("the shape of trX is "+ str(trX.shape))
tsX = data['tsX']
print("the shape of tsX is "+ str(tsX.shape))
trY = data['trY']
print("the shape of trY is "+ str(trY.shape))
tsY = data['tsY']
print("the shape of tsY is "+ str(tsY.shape))


#convert them to numpy arrays
np.asarray(trX)
np.asarray(trY)
np.asarray(tsX)
np.asarray(tsY)


#seperate data by class
x7 = []
for count, i in enumerate(trY[0]):
    if i == 0:
        x7.append(trX[count])
x8 = []
for count, i in enumerate(trY[0]):
    if i == 1:
        x8.append(trX[count])

#calculate the means and standard deviations for each image
means = np.mean(trX, axis = 1)
stdev = np.std(trX, axis = 1)

#Build an intercept of size equal to dataset
intercept = np.ones(means.shape)

print(intercept.shape, means.shape)
#output: ((12116,), (12116,))

#build the features set by horizontally stacking intercept, mean and standard deviation
features = [intercept, means,stdev]
#convert the features array into numpy array for simpler matrix calculations
features = np.asarray(features)
features = features.T

print(features.shape)
#output: (12116, 3)

print(trX.shape[1])
#output: 784

#define the sigmoid function based on formula
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#define gradient_ascent
def gradient_ascent(x, y, predictions, learning_rate,weights):
    error = y-predictions
    gradient = (np.dot(x.T, error))/(x.shape[0])
    weights += learning_rate*gradient
    return weights

#build logistic regression
def logistic_regression(x,y, learning_rate, steps):
    weights = np.zeros(features.shape[1])
    for i in range(steps):
        predictions = sigmoid(np.dot(x, weights))
        gradient_ascent(x,y,predictions, learning_rate,weights)
    return weights

#get the model which contains the weights
model = logistic_regression(features, trY[0], learning_rate = 0.1, steps = 250000)

print(model)
#array([-3.11031542, 30.28442177, -3.03516435])

#calculate the means and standard deviation for each image in test dataset
test_means = np.mean(tsX, axis = 1)
test_stdev = np.std(tsX, axis = 1)
#Build the intercept of size equal to test data
test_intercept = np.ones(test_means.shape)
#build the test features array by horizontally stacking intercept, means and standard deviations
testFeatures = [test_intercept, test_means, test_stdev]
#convert the test features array into a numpy array
testFeatures = np.asarray(testFeatures)
testFeatures = testFeatures.T
#Get the predictions for images based on the model
values = np.dot(testFeatures,model)
values = sigmoid(values)

#classify the image based on sigmoid values into class 7 or class 8
output = []
for i in values:
    if i < 0.5:
        output.append(0.0)
    else:
        output.append(1.0)


#define overall Accuracy
def overallAccuracy(prediction, groundTruth):
	correct = 0
	for count, i in enumerate(output):
		if i == tsY[0][count]:
			correct+=1
	overallAccuracy = correct/(tsY[0].shape)[0]
	return overallAccuracy

#define individual class accuracies
def classAccuracies(prediction, groundTruth):
    count7right = 0
    count8right = 0
    count7 = 0
    count8 = 0
    for i in groundTruth:
        if (i == 0.0):
            count7+=1
        elif (i == 1.0):
            count8 += 1
    for i in zip(prediction, groundTruth):
        if(i[0]==i[1]==0.0):
            count7right += 1
        elif (i[0]==i[1]==1.0):
            count8right += 1
    return [count7right/count7, count8right/count8]

#calculate the overalla and individual class accuracies
overallAccuracy(output, tsY[0])
classAccuracies(output, tsY[0])

print(overallAccuracy)
#output: 0.7132867132867133

print(classAccuracies)
#output: [0.7607003891050583, 0.6632443531827515]
