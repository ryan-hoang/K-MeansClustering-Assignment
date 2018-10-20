import numpy as np
import random
import scipy.spatial.distance as scipy
import scipy.sparse as sc
from sklearn.preprocessing import normalize

def getDistances(X,cent):
    
    cent = [cent]

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
    print(np.array(dist))

    print("#################")
    dist = []
    temp = []
    for x in X:
        for c in cent:
            print(cent)
            print("subtract: ",x , c ,np.subtract(x,c))
            temp.append(np.linalg.norm(np.subtract(x,c))**2)#1 - (np.dot(x,c) / (np.linalg.norm(x) * np.linalg.norm(c)))
        #print("\n",temp)    
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
    

    
    c = vector3
    x = []
    x.append(vector1)
    x.append(vector2)
    x.append(vector3)
    
    #print(vector1, np.linalg.norm(vector1), np.linalg.norm(np.linalg.norm(vector1)))
    vector1 = np.array(vector1)
    
    vector2 = np.array(vector2)
    distnorm = 1 - (np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    #dist = scipy.cosine(vector1,vector2)
    
    
    getDistances(np.array(x),np.array(c))
    

