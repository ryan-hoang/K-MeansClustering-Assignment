import numpy as np
import sys
import random
import pickle
import scipy.sparse as sc
import time
import warnings
from sklearn.decomposition import *
from sklearn.pipeline import *
from sklearn.feature_selection import *
import os
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.metrics import v_measure_score

def cluster(vectors, k,centroids):
    clusterLabels = [-1]*(len(vectors))
    #print(clusterLabels)
    for count in range(len(vectors)):
        x = vectors[count].toarray()[0]
        #print(x)
        dists = []
        #print("centroids: {} iteration:{}", centroids, count)
        for c in centroids:
            print("Assigning cluster for vector {}".format(count))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
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

    for i in range(len(vectors)): # populate list of clusters with vectors
        #print(type(vectors.getrow(i).toarray()), vectors.getrow(i).toarray()[0])
        listofclusters[clusterLabels[i]-1].append(vectors[i].toarray()[0])

    updatedCentroids = []
    #print(len(listofclusters))
    print("Updating Centroids")
    for cluster in listofclusters: # calculate mean vector in each cluster set that to the centroid
        updatedCentroids.append(np.mean(cluster, axis=0))
    print("Done Updating Centroids")
    return clusterLabels,np.array(updatedCentroids)

def initCentroids(vectors, k):
    centroids = []
    randomnumber = random.randint(0,len(vectors))
    '''if(len(vectors) == 8580):
        randomnumber = 4761  # this is the number that generated my highest score, I didnt seed my random so im just manually setting it
    '''#print(randomnumber)
    t = vectors[randomnumber]
    with open("startingvectorforlastrun.txt", 'w') as startvectnum:
        startvectnum.write("{}\n".format(randomnumber))
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
    cen = sc.lil_matrix(cent)
    dist = []
    temp = []
    for indx in range(len(vectors)):
        x = vectors[indx].toarray()[0]
        for indc in range(cen.get_shape()[0]):
            print("Calculating Distance from x: {} to Centroid: {}".format(indx,indc))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
            c = cen[indc,0:].toarray()
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
        inputDataProcessed = []
        for ind,line in enumerate(inputDataRaw):
            temp = sc.dok_matrix((1,126373),dtype=np.int8)
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
                temp[0,index-1] = count
            inputDataProcessed.append(temp)

        aggregatedVectors = sc.vstack(inputDataProcessed,format = "dok")
        transformed = None
        nextTransform = None
        tsvd = None
        
        if os.path.exists('tsvdText.dat'):
            transformed = pickle.load(open("tsvdText.dat",'rb'))
            print("Reduced dimension data loaded.")
        else:
            tsvd = TruncatedSVD(n_components=2000)
            transformed = tsvd.fit_transform(aggregatedVectors)
            pickle.dump(transformed,open("tsvdText.dat",'wb'))
            transformed = pickle.load(open("tsvdText.dat",'rb'))
        '''
        if os.path.exists('varThresholdText.dat'):
            nextTransform = pickle.load(open("varThresholdText.dat",'rb'))
            print("data with low variance removed loaded")
        else:
            selector = VarianceThreshold()
            print("removing features with low variance")
            nextTransform = selector.fit_transform(transformed)
            pickle.dump(nextTransform,open("varThresholdText.dat",'wb'))
            nextTransform = pickle.load(open("varThresholdText.dat",'rb'))
        '''
        '''    
        explainedvariance = tsvd.explained_variance_ratio_.cumsum()
        with open("explainedvariance.txt",'w') as variances:
            for i,var in enumerate(explainedvariance):
                variances.write("Variance for {} components: {}\n".format(i+1,var))    
        '''
        n = transformed.tolist()
        inputDataProcessed = []
        for vect in n:
            inputDataProcessed.append(sc.csr_matrix(vect).todok())
    
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataRaw)))
        #pickle.dump(inputDataProcessed,open("textCSRSparse.dat",'wb'))
    elif mode == "iris":
        inputDataProcessed = []
        for row,line in enumerate(inputDataRaw):
            temp = sc.dok_matrix((1,4),dtype=np.float64)
            splitLine = line.split()
            for col,item in enumerate(splitLine):
                #print(item,type(float(item)))
                temp[0,col] = float(item)
            inputDataProcessed.append(temp)
        print("Data loaded. {} data vectors loaded.\n".format(len(inputDataRaw)))
    else:
        print("No mode specified. Exiting.")
        sys.exit(1)
        #print(inputDataProcessed.toarray())
    return inputDataProcessed

def generateAnswerFile(labels):
    with open("clusterLabels.data", 'w') as outFile:
        for item in labels:
            outFile.write("{}\n".format(item))
# start Main------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    try:
        k = int(sys.argv[2])
    except:
        k = 3
    
    maxiterations = 50

    for i in range(3,21,2):
        k=i
        start = time.time()    
        vectors = processData()
        centroids = initCentroids(vectors,k)
        print("Initial Centroids Calculated\n")
        iterations = 0
        
        with open("resultFile.txt", 'a') as resultFile:
            while True:
                iterations = iterations + 1
                print("Current Iteration: ", iterations)
                assignments,newCentroids = cluster(vectors,k,centroids)
                #print("Old Centroids: {}\nNew Centroids: {}".format(centroids,newCentroids))
                if(np.array_equal(centroids,newCentroids) or iterations == maxiterations):
                    generateAnswerFile(assignments)
                    densevects = []
                    for vect in vectors:
                        densevects.append(vect.toarray()[0])
                    coeff = metrics.silhouette_score(densevects,assignments,metric='euclidean')
                    resultFile.write("Silhouette Coefficient for k={}: {}\n".format(k,coeff))
                    break;
                else:
                    centroids = newCentroids
            
        end = time.time()
        elapsedTime = (end-start)/60
        print("\nElapsed Time: {} Total Iterations: {}".format(elapsedTime,iterations))#,iterations
        
'''        x = []
        y = []
        with open("clusterLabels.data",'r') as clusters:
            for line in clusters:
                x.append(line.split()[0])
                
        with open("clusterLabelsSCORE58.data",'r') as clus:
            for line in clus:
                y.append(line.split()[0])
                
        if(v_measure_score(x,y)>=90.0):
            break;
        '''
