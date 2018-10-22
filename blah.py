from sklearn.metrics import v_measure_score

x = []
y = []
with open("clusterLabels.data",'r') as clusters:
    for line in clusters:
        x.append(line.split()[0])
        
with open("clusterLabelsSCORE58.data",'r') as clus:
    for line in clus:
        y.append(line.split()[0])
        
print(v_measure_score(x,y))