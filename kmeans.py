import numpy as np
import sys
import random
import pickle
import scipy.sparse as sc

def cluster(vectors, k,centroids):
    vectors = np.array(vectors)
    clusterLabels = [-1]*len(vectors)
    #print(clusterLabels)
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
    
    listofclusters = []
    for i in range(0,k):
        listofclusters.append([])
    
    for i in range(len(vectors)): # populate list of clusters with vectors 
        listofclusters[clusterLabels[i]-1].append(vectors[i])
    
    updatedCentroids = []
    print(len(listofclusters))
    for cluster in listofclusters: # calculate mean vector in each cluster set that to the centroid
        updatedCentroids.append(np.mean(cluster, axis=0))
    
    return clusterLabels,np.array(updatedCentroids)
    
def initCentroids(vectors, k):
    centroids = []
    centroids.append(np.array(random.sample(vectors,1)[0]))
    vectors = np.array(vectors)
    for i in range(1,k):
        squaredDists = getDistances(vectors, centroids)
        rand = random.random()
        cumulativeProbabilities = (squaredDists/squaredDists.sum()).cumsum()
        centroids.append(vectors[np.where(cumulativeProbabilities >= rand)[0][0]])
    return np.array(centroids)
    
def getDistances(X,cent):
    vectors = sc.csr_matrix(X,copy = True)
    cen = sc.csr_matrix(cent,copy=True)
    dist = []
    temp = []
    for indx in range(vectors.get_shape()[0]):
        x = vectors.getrow(indx).toarray()[0]
        for indc in range(cen.get_shape()[0]):
            c = cen.getrow(indc).toarray()[0]
            temp.append(np.linalg.norm(np.subtract(x,c))**2)
        dist.append(np.array(temp).min())
        temp = []

    return np.array(dist)
    
def processData():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
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
        inputDataProcessed = sc.csr_matrix((8550,126373),dtype=np.int8)
        for ind,line in enumerate(inputDataRaw):
            splitLine = line.split()
            iterations = int(len(splitLine)/2)
            #vector = [0] * 126373
            print(ind)
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
            for i in range(iterations):
                index = int(splitLine[0])
                count = int(splitLine[1])
                del splitLine[0]
                del splitLine[0]
                inputDataProcessed[ind,index-1] = count
            #inputDataProcessed.append(vector)
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataProcessed)))
        pickle.dump(inputDataProcessed,open("textCSRSparse.dat",'wb'))
    else:
        for line in inputDataRaw:
            splitLine = line.split()
            vector = []
            for item in splitLine:
                vector.append(float(item))
            inputDataProcessed.append(np.array(vector))
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataProcessed)))
    return inputDataProcessed
    
def generateAnswerFile(labels):
    with open("clusterLabels.data", 'w') as outFile:
        for item in labels:
            outFile.write("{}\n".format(item))
# start Main------------------------------------------------------------
if __name__ == '__main__':
    k = 3
    vectors = processData()
    centroids = initCentroids(vectors,k)
    
    while True:
        assignments,newCentroids = cluster(vectors,k,centroids)
        if(np.array_equal(centroids,newCentroids)):
            generateAnswerFile(assignments)
            break;
        else:
            centroids = newCentroids
