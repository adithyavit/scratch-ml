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



def Mean(data):
    return np.mean(data,axis=1)
#calculate means for entire dataset
DataMean = Mean(trX)

#calculate means for each class
class7Mean = Mean(x7)
class8Mean = Mean(x8)

def Stdev(data):
    return np.std(data,axis=1)

#calculate standard deviations for entire dataset
DataStdev = Stdev(trX)

#calculate standard deviations for each class
class7std = Stdev(x7)
class8std = Stdev(x8)

#calculate class probabilities using gaussian function
def GaussProb(x,mean,stdev):
    prob = (1/(stdev*(math.sqrt(2*math.pi))))*(math.exp(-0.5*((x-mean)/stdev)**2)) 
    return prob

print(GaussProb(tsX[0][0],class7Mean[0], class7std[0]))
#output: 1.2047399538597614

def probClass7(inputvec):
    ans = 1
    for count, i in enumerate(inputvec):
        ans *= GaussProb(i,class7Mean[count],class7std[count])
    return ans
def probClass8(inputvec):
    ans = 1
    for count, i in enumerate(inputvec):
        ans *= GaussProb(i,class8Mean[count],class8std[count])
    return ans

def testDataPrediction(tsX):
    ans = []
    for i in tsX:
        prob7 = probClass7(i)
        #print(prob7)
        prob8 = probClass8(i)
        #print(prob8)
        if prob7 > prob8:
            ans.append(0.0)
        else:
            ans.append(1.0)
    return ans

#give mean and std as two featuers instead of entire pixels
tsX = [np.mean(tsX, axis=1),np.std(tsX, axis=1)]

finalAnswer = testDataPrediction(tsX)

#define overall Accuracy
def overallAccuracy(prediction, groundTruth):
	correct = 0
	for count, i in enumerate(output):
		if i == tsY[0][count]:
			correct+=1
	overallAccuracy = correct/(tsY[0].shape)[0]
	return overallAccuracy


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

overallAccuracy(finalAnswer, tsY[0])
#output: 0.7032967032967034
classAccuracies(finalAnswer, tsY[0])
#output: [0.7402723735408561, 0.6642710472279261]
