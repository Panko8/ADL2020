from matplotlib import pyplot as plt
import numpy as np
import torch

def Z(x,a=1):
    x=torch.tensor([x]).float()
    return (-0.499+a*torch.sigmoid(x*4/20))*50

asymptote = Z(20) 

fig, axes = plt.subplots()

x=np.arange(1,30,0.01)
y=asymptote/np.array(list(map(Z,x)))
axes.plot(x,y)

def Z(x,a=1.5):
    x=torch.tensor([x]).float()
    return (-0.499+a*torch.sigmoid(x*4/20))*50

x=np.arange(1,30,0.01)
y=asymptote/np.array(list(map(Z,x)))
axes.plot(x,y)



plt.ylabel("Magnification")
plt.xlabel("Ground Truth Distance")
plt.title("customMSE penalizes more on short distance")
plt.xlim(0,32)
plt.ylim(0,10)
axes.yaxis.set_ticks(np.arange(0, 10.1, 1))
plt.show()

