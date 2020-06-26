from matplotlib import pyplot as plt
import numpy as np
import torch

def Z(x,a=1,b=4/20,c=50,d=-0.499):
    x=torch.tensor([x]).float()
    return (d+a*torch.sigmoid(x*b))*c


fig, axes = plt.subplots()

asymptote = Z(20) 
x=np.arange(0,30,0.01)
y=asymptote/np.array([Z(i) for i in x])
axes.plot(x,y)

#leg = fig.legend(labels=["c={}".format(c) for c in C], bbox_to_anchor=[0.9,0.88], fontsize=8)
plt.ylabel("Magnification")
plt.xlabel("Ground Truth Distance (meter)")
plt.title("customMSE penalizes more on short distance")
plt.xlim(0,32)
plt.ylim(0,10)
axes.yaxis.set_ticks(np.arange(0, 10.1, 1))
plt.show()

