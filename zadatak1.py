import numpy as np
import matplotlib . pyplot as plt

x= np.array([1,2,3,3,1])
y= np.array([1,2,2,1,1]) 

plt.plot(x,y,'b',marker="X",linewidth=5,color="green")
plt.axis([0,4,0,4])

plt . xlabel ('X')
plt . ylabel ('Y')
plt . title ( 'Prvi Zad')
plt . show ()