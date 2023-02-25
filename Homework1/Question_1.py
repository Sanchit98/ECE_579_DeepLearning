import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

train_points = np.array([[0,1,0], [0,1,1], [1,2,1], [1,2,0], [1,2,2], [2,2,2], [1,2,-1], [2,2,3], [-1,-1,-1], [0,-1,-2], [0,-1,1], [-1,-2,1]])
train_labels = np.array([0,0,0,0,1,1,1,1,2,2,2,2])
test = np.array([[1,0,1]])

labels_mapping = {0: 'A', 1: 'B', 2: 'C'}

# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################
    for pt in newInput:
        distance=np.linalg.norm(dataSet-pt,axis=1)
        closest_indices=np.argsort(distance)[:k]          #Obtain the indices of the k nearest neighbours
        closest_labels=[train_labels[i] for i in closest_indices]  #Obtain the labels of the k nearest neighbours
        result_label = labels_mapping[np.argmax(np.bincount(closest_labels))]
        result.append(result_label)
    
    ####################
    # End of your code #
    ####################
    return result

outputlabel_k1 = kNNClassify(test,train_points,train_labels,1)
outputlabel_k2 = kNNClassify(test,train_points,train_labels,2)
outputlabel_k3 = kNNClassify(test,train_points,train_labels,3)

print('KNN classfied label for K=1:', outputlabel_k1)
print('KNN classfied label for K=2:', outputlabel_k2)
print('KNN classfied label for K=3:', outputlabel_k3)

