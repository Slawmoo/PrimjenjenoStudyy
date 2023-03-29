import numpy as np
import matplotlib . pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),delimiter=",", skiprows=1)

print("\nSvi:")
print(min(data[:,0]))
print(max(data[:,0]))
print(sum(data[:,0])/len(data[:,0]))

plt.scatter(data[:,0],data[:,3],c='g',s=data[:,5]*50)

for i in range(2,32):
    plt.text(data[i,0], data[i,3], s=str(data[i,5]), fontsize=10)

idx = data[:,1]== 6
data_6 = data[idx,:]

print("\nSamo 6 cilindara:")
print(min(data_6[:,0]))
print(max(data_6[:,0]))
print(sum(data_6[:,0])/len(data_6[:,0]))

plt.xlabel ('X')
plt.ylabel ('Y')
plt.title ( 'Drugi Zad')
plt.autoscale()
plt.show ()

