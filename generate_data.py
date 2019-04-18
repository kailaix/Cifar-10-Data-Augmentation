#%%
from cifar10 import CIFAR10
c = CIFAR10()
c.data_augmentation(10000)

#%%
from scipy import io as io 
import numpy as np 
c.y_train = np.argmax(c.y_train, axis=1).astype("int32")
c.y_test = np.argmax(c.y_test, axis=1).astype("int32")
io.savemat("train.mat",{"x":c.x_train, "y":c.y_train})
io.savemat("test.mat",{"x":c.x_test, "y":c.y_test})
