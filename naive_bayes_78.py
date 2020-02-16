import numpy as np # linear algebra
import math
from scipy.io import loadmat

#load the data
data= loadmat('mnist_data.mat') 


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

def Stdev(data):
    return np.std(data,axis=1)

#calculate means for each class
class7f1 = Mean(x7)
class8f1 = Mean(x8)

#calculate standard deviations for each class
class7f2 = Stdev(x7)
class8f2 = Stdev(x8)


class7f1_mean = class7f1.mean()
class7f1_std = class7f1.std()
class8f1_mean = class8f1.mean()
class8f1_std = class8f1.std()

class7f2_mean = class7f2.mean()
class7f2_std = class7f2.std()
class8f2_mean = class8f2.mean()
class8f2_std = class8f2.std()

#calculate class probabilities using gaussian function
def GaussProb(x,mean,stdev):
    prob = (1/(stdev*(math.sqrt(2*math.pi))))*(math.exp(-0.5*((x-mean)/stdev)**2)) 
    return prob

def probClass7(i):
    ans = GaussProb(i[0],class7f1_mean,class7f1_std)*GaussProb(i[1],class7f2_mean,class7f2_std)
    return ans
def probClass8(i):
    ans = GaussProb(i[0],class8f1_mean,class8f1_std)*GaussProb(i[1],class8f2_mean,class8f2_std)
    return ans

def testDataPrediction(ts_features):
    ans = []
    for i in ts_features:
        if1 = i[0]
        if2 = i[1]
        prob7 = probClass7(i)*(len(x7)/len(x7)+len(x8))
        prob8 = probClass8(i)*(len(x8)/len(x7)+len(x8))

        if prob7 > prob8:
            ans.append(0.0)
        else:
            ans.append(1.0)
    return ans

print(GaussProb(tsX[0][0],class7f1_mean,class7f1_std))


ts_features =  [tsX.mean(1), tsX.std(1)]
ts_features = np.transpose(ts_features)

prediction = testDataPrediction(ts_features)

groundTruth = tsY

#define overall Accuracy
def overallAccuracy(prediction, groundTruth):
	correct = 0
	for count, i in enumerate(prediction):
		if i == tsY[0][count]:
			correct+=1
	overallAccuracy = correct/(tsY[0].shape)[0]
	return overallAccuracy

overallAccuracy(prediction,groundTruth)

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



classAccuracies(prediction, tsY[0])

overallAccuracy(prediction, tsY[0])
