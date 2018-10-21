import numpy as np
import sys
import random
import pickle
import scipy.sparse as sc

def cluster(vectors, k,centroids):
    clusterLabels = [-1]*(vectors.get_shape()[0])
    #print(clusterLabels)
    for count in range(vectors.get_shape()[0]):
        x = vectors[count].toarray()[0]
        dists = []
        #print("centroids: {} iteration:{}", centroids, count)
        for c in centroids:
            dists.append(1 - (np.dot(x,c) / (np.linalg.norm(x) * np.linalg.norm(c)))) # list of distances from x to every centroid (cosine distance)
        min = 0
        for i in range(k):
            #print(i)
            if dists[i] < dists[min]:
                min = i
        clusterLabels[count] = min+1

    #recalculate clusters now

    listofclusters = []
    for i in range(0,k):
        listofclusters.append([])

    for i in range(vectors.get_shape()[0]): # populate list of clusters with vectors
        #print(type(vectors.getrow(i).toarray()), vectors.getrow(i).toarray()[0])
        listofclusters[clusterLabels[i]-1].append(vectors[i].toarray()[0])

    updatedCentroids = []
    #print(len(listofclusters))
    for cluster in listofclusters: # calculate mean vector in each cluster set that to the centroid
        updatedCentroids.append(np.mean(cluster, axis=0))

    return clusterLabels,np.array(updatedCentroids)

def initCentroids(vectors, k):
    centroids = []
    randomnumber = random.randint(0,vectors.get_shape()[0])
    #print(randomnumber)
    t = vectors[randomnumber,0:]
    #print("t: ",t.toarray())
    centroids.append(t.toarray()[0].tolist())
    #print("Centroids:", centroids)
    print("First Centroid Calculated.\n")

    for i in range(1,k):
        print("Calculating Initial Centroid {}/{}".format(i,k))
        #print("Centroids:{} Iteration: {}".format(centroids,i))
        squaredDists = getDistances(vectors, centroids)
        rand = random.random()
        cumulativeProbabilities = (squaredDists/squaredDists.sum()).cumsum()
        blah = vectors[np.where(cumulativeProbabilities >= rand)[0][0]].toarray()
        #print(type(blah))
        centroids.append(blah.tolist()[0])
    #print("Centroids returned by initCentroids:", centroids)
    return centroids

def getDistances(X,cent):
    vectors = X
    #print("Cent in getDistances:", cent)
    cen = sc.dok_matrix(cent)
    dist = []
    temp = []
    for indx in range(vectors.get_shape()[0]):
        x = vectors[indx].toarray()[0]
        for indc in range(cen.get_shape()[0]):
            print("Calculating Distance from x: {} to Centroid: {}".format(indx,indc))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
            c = cen[indc].toarray()
            #print(x,"c: ",c)
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
        inputDataProcessed = sc.dok_matrix((8580,126373),dtype=np.int8)
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
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataRaw)))
        pickle.dump(inputDataProcessed,open("textCSRSparse.dat",'wb'))
    else:
        inputDataProcessed = sc.dok_matrix((150,4),dtype=np.float64)
        for row,line in enumerate(inputDataRaw):
            splitLine = line.split()
            for col,item in enumerate(splitLine):
                #print(item,type(float(item)))
                inputDataProcessed[row, col] = float(item)
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataRaw)))
        #print(inputDataProcessed.toarray())
    return inputDataProcessed

def generateAnswerFile(labels):
    with open("clusterLabels.data", 'w') as outFile:
        for item in labels:
            outFile.write("{}\n".format(item))
# start Main------------------------------------------------------------
if __name__ == '__main__':
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    k = 7
    vectors = processData()
    centroids = initCentroids(vectors,k)
    print("Initial Centroids Calculated\n")

    while True:
        assignments,newCentroids = cluster(vectors,k,centroids)
        print("Old Centroids: {}\nNew Centroids: {}".format(centroids,newCentroids))
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)
        if(np.array_equal(centroids,newCentroids)):
            generateAnswerFile(assignments)
            break;
        else:
            centroids = newCentroids
