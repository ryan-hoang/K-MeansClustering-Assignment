import numpy as np
import sys
import random

# start Main------------------------------------------------------------
if __name__ == '__main__':
    k = 3
    vectors = processData()
    centroids = initCentroids(vectors,k)
    
    while True:
        assignments,newCentroids = cluster(vectors,k,centroids)
        if(centroids == newCentroids):
            break;
        else:
            centroids = newCentroids
    
def cluster(vectors, k,centroids):
    clusterLabels = [-1]*len(vectors)
    for count,x in enumerate(vectors):
        dists = []
        for c in centroids:
            dists.append(1 - (np.dot(x,c) / (np.linalg.norm(x) * np.linalg.norm(c)))) # list of distances from x to every centroid (cosine distance)
        min = 0
        for i in range(k):
            if dists[i] < dists[min]:
                min = i
        clusterLabels[count] = min+1
    
    #recalculate clusters now
    
    
    return clusterLabels,centroids
    
def initCentroids(vectors, k):
    centroids = []
    centroids.append(random.sample(vectors,1))
    for i in range(k):
        squaredDists = getDistances(vectors, centroids)
        rand = random.random()
        cumulativeProbabilities = (squaredDists/squaredDists.sum()).cumsum()
        centroids.append(vectors[np.where(cumulativeProbabilities >= rand)[0][0]])
    return centroids
    
def getDistances(X,cent):
    dist = []
    temp = []
    for x in X:
        for c in cent:
            temp.append(np.linalg.norm(x-c)**2)
        dist.append(np.array(temp).min())
        temp = []

    return np.array(dist)
    
def processData():
    try:
        mode = sys.argv[1]
    except:
        mode = "iris"
    if mode == "text":
        inputFile = "input.dat"
    else:
        mode = "iris"
        inputFile = "iris.data"

    print("Mode: {}".format(mode))
    inputDataRaw = []
    inputDataProcessed = []
    
    with open(inputFile) as file:
        for line in file:
            inputDataRaw.append(line) # each line of input is one element of inputDataRaw
    
    if mode == "text":
        for line in inputDataRaw:
            splitLine = line.split()
            iterations = int(len(splitLine)/2)
            vector = [0] * 126373
            for i in range(iterations):
                index = int(splitLine[0])
                count = int(splitLine[1])
                del splitLine[0]
                del splitLine[0]
                vector[index-1] = count
            inputDataProcessed.append(vector)
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataProcessed)))
    else:
        for line in inputDataRaw:
            splitLine = line.split()
            vector = []
            for item in splitLine:
                vector.append(float(item))
            inputDataProcessed.append(vector)
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataProcessed)))
    return np.array(inputDataProcessed)