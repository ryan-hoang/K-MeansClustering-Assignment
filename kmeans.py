import numpy as np
import sys

# start Main------------------------------------------------------------
if __name__ == '__main__':
    
    mode = sys.argv[1]
    
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
            iterations = len(splitLine)/2
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
    
    print("Data:\n {}".format(inputDataProcessed))
    