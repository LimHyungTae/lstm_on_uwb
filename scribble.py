import numpy as np
a = np.random.randint(0,5,(10,5,2))
print (a)
b = a[:,-1,:]

print (b)
print (a.shape, b.shape)