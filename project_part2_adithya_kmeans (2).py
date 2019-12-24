#importing pandas for dataframe manipulation
import pandas as pd
#importing numpy for vector calculations
import numpy as np
#importing scipy for distance and data loading function
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.spatial.distance import cdist
#importing random to select random variables
import random
#import matplotlib for objective functino visulization.
import matplotlib.pyplot as plt

#setting a seed number for reproducable results.
random.seed(42)

#declaring the image path
x = 'AllSamples.mat'
#load the data from the path using loadmat function
data = loadmat(x)

#get the data part from the mat file leaving other unrequired things.
X = data['AllSamples']

#convert the input into numpy array
X = np.asarray(X)
X = np.around(X,decimals=2)

#store the shape of input into a variable named N
N = X.shape[0]

#create an empty array that will later store the cluster number for each point
cluster_array = np.reshape(np.zeros(N), (-1, 1))
#combine the input data with cluter number array
array_with_cluster_num = np.concatenate((X,cluster_array),axis=1)

#define the random initialization function
def random_initialize(X,c):
    #let array of cluster centers be c_ar
    c_ar = np.zeros((c, 2))
    #let no of features be N
    N = X.shape[0]
    #so we need to generate two random points to act as cluster centers
    for index, sample in enumerate(c_ar):
        c_ar[index] = X[np.random.choice(N)]
    return c_ar

#clutering algorithm that assigns each point to a cluster
def cluster(X,c_ar):
    #find the distance from each point to every cluster
    dist = cdist(X,c_ar)
    dist = np.asarray(dist)
    #find the nearest cluster for each point
    for index,i in enumerate(dist):
        idx = np.argmin(dist[index])
        array_with_cluster_num[index][-1] = idx+1
    
    return array_with_cluster_num

#function to find the cluster centers
def cluster_centers2(arr,c):
    #initalize the c_sum and c_vectors
    c_sums = []
    c_means = []
    #for each cluster, append the sum of all points in the cluster
    for j in range(c):
        c_sums.append(sum(arr[i] for i in range(len(arr)) if arr[i][2] == j+1))
    #for each cluster, find the c_mean from the c_sums
    for j in range(c):
        c_means = c_sums[j]/c_sums[j][2]*(j+1)
    #convert the c_means array into numpy array
    c_ar = np.asarray(c_means)
\
    c_ar = c_ar[:,:c:]
    return c_ar

#definition of farthest from the previous
def furthest_from_first(X,c_ar,c):
    c_ar = []
    c_ar.append(X[np.random.choice(N)])
    #for each cluster, create an empy array
    for j in range(1,c):
        k = []
    #for each point in X, find the distance from all the previous place
        for i in range(len(X)):
            dist = 0
            size=0
            for ind in range(len(c_ar)):
                size+=1
                dist += distance.euclidean(X[i],c_ar[ind])
            k.append(dist/size)
        #Get the maximum value from all the disances
        q = max(k)
        m = k.index(q)
        c_ar.append(X[m])
    return c_ar

def objective_function(arr,c_ar,c):
    m_sqr = 0
    dists=[]
    #create an empty array of distances.
    for i in range(int(c)):
        dists.append([])
    #find the objective function
    for i in arr:
        k = int(i[2])-1
        dst = distance.euclidean(c_ar[k],i[0:2])
        dists[k].append(dst)
    obj=0
    for i in range(c):
        obj+=sum(dists[i])
    return obj
#clusters the data using cluster and cluster_centers definitons 
def clustering_center(X,c_ar,c):
    arr = cluster(X,c_ar)
    c_ar = cluster_centers(arr,c)
    prev = c_ar
    arr = cluster(arr[:,:2:],c_ar)
    c_ar = cluster_centers(arr,c)
    while ((prev == c_ar).all()):
        prev = c_ar
        c_ar = cluster_centers(arr,c)
        arr = cluster(arr[:,:2:],c_ar)
    return objective_function(arr,c_ar,c)

def clustering_for_multiple_k_vals(X,c):
    #np.random.seed(91)
    c_ar = np.zeros((c, 2))
    #let no of features be N
    N = X.shape[0]
    #so we need to generate two random points to act as cluster centers
    for index, sample in enumerate(c_ar):
        c_ar[index] = X[np.random.choice(N)]
    #create an empty array that will later store the cluster number for each point
    cluster_array = np.reshape(np.zeros(N), (-1, 1))
    #combine the input data with cluter number array
    array_with_cluster_num = np.concatenate((X,cluster_array),axis=1)
    #run the actual clustering algorithm
    k = clustering_center(X,c_ar,c)
    return k

#running the clustring algorithm with random initilization
objs = []
c_vals = []
for c in range(2,11):
    c_ar = random_initialize(X,c)
    c_vals.append(c)
    k = clustering_for_multiple_k_vals(X,c)
    objs.append(k)
    #print(k)
plt.plot(c_vals,objs,label="objectve_function vs number of clusters")

#running the clustering algorithm with farthest from the first algorithm.
objs = []
c_vals = []
for c in range(2,11):
    c_ar = furthest_from_first(X,c_ar,c)
    #print(c)
    c_vals.append(c)
    k = clustering_for_multiple_k_vals(X,c)
    objs.append(k)
plt.plot(c_vals,objs,label="objectve_function vs number of clusters")