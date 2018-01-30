import matplotlib

from matplotlib import pyplot as plt
def images(X):
X = X - X.min(); X = X / X.max()
X = X.reshape([5,10,32,32,3]).transpose([0,2,1,3,4]).reshape([160,320,3])
plt.figure(figsize=(8,4))
plt.imshow(X,cmap=’seismic’,vmin=-1,vmax=1)
def heatmaps(H):
H = H / H.max()
H = H.reshape([5,10,32,32]).transpose([0,2,1,3]).reshape([160,320])
plt.figure(figsize=(8,4))
plt.imshow(H,cmap=’seismic’,vmin=-1,vmax=1)
