from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

dataX1 = generate_data(500, 1)
dataX2 = generate_data(500, 2)
dataX3 = generate_data(500, 3)
dataX4 = generate_data(500, 4)
dataX5 = generate_data(500, 5)

matrices = [dataX1,dataX2,dataX3,dataX4,dataX5]
# plt.scatter(dataX1[:,0], dataX1[:,1])
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter plot')
# plt.show()

# plt.scatter(dataX2[:,0], dataX2[:,1])
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter plot')
# plt.show()

# plt.scatter(dataX3[:,0], dataX3[:,1])
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter plot')
# plt.show()

# plt.scatter(dataX4[:,0], dataX4[:,1])
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter plot')
# plt.show()

# plt.scatter(dataX5[:,0], dataX5[:,1])
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter plot')
# plt.show()

numCentara = range(1, 10)
for matrix in matrices:
    # Loop over the range of number of clusters and fit KMeans model
    inertias = []
    for k in numCentara:
        model = KMeans(n_clusters=k, n_init='auto')
        model.fit(matrix)
        inertias.append(model.inertia_)

    # Create a line plot of the inertia values
    plt.plot(numCentara, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of Clusters')    
plt.show()
