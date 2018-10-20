import numpy as np
import random
import scipy.spatial.distance as scipy

def getDistances(X,cent):
    distances = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
    print(distances)
    dist = []
    temp = []
    for x in X:
        for c in cent:
            temp.append(np.linalg.norm(x-c)**2)#1 - (np.dot(x,c) / (np.linalg.norm(x) * np.linalg.norm(c)))
        dist.append(np.array(temp).min())
        temp = []
        
    print(np.array(dist))

if __name__ == '__main__':
    vector1 = [1,2,3,4,5,6]
    vector2 = [3,5,4,7,9,11]
    vector3 = [12,4,66,22,25,10]
    
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    vector3 = np.array(vector3)
    
    x = []
    x.append(vector1)
    x.append(vector2)
    x.append(vector3)
    x = np.array(x)
    
    print(np.mean(x, axis=0))
    
    '''
    c = vector3
    x = []
    x.append(vector1)
    x.append(vector2)
    x.append(vector3)
    #print(vector1, np.linalg.norm(vector1), np.linalg.norm(np.linalg.norm(vector1)))
    vector1 = np.linalg.norm(np.array(vector1))
    
    vector2 = np.linalg.norm(np.array(vector2))
    distnorm = 1 - (np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    dist = scipy.cosine(vector1,vector2)
    
    
    getDistances(np.array(x),np.array(c))
    '''

