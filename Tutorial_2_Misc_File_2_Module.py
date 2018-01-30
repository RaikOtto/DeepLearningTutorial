# http://heatmapping.org/wifs2017/modules.pdf

import numpy
# -------------------------
# Feed-forward network
# -------------------------
class Network:
  def __init__(self,layers):
    self.layers = layers
    
  def forward(self,Z):
    for l in self.layers: Z = l.forward(Z)
    return Z
  
  def gradprop(self,DZ):
    for l in self.layers[::-1]: DZ = l.gradprop(DZ)
      return DZ
  def update(self,lr):
    for l in self.layers: l.update(lr)
  def dump(self):
    for l in self.layers: l.dump()
  
# -------------------------
# ReLU activation layer
# -------------------------
class ReLU:
  def forward(self,X): self.Z = X>0; return X*self.Z
  def gradprop(self,DY): return DY*self.Z
  def update(self,lr): pass
  def dump(self): pass
# -------------------------
# Sum-pooling layer
# -------------------------
class Pooling:
  def forward(self,X):
    self.X = X
    self.Y = 0.5*(X[:,::2,::2,:]+X[:,::2,1::2,:]+X[:,1::2,::2,:]+X[:,1::2,1::2,:])
    return self.Y

  def gradprop(self,DY):
    self.DY = DY
     DX = self.X*0
  for i,j in [(0,0),(0,1),(1,0),(1,1)]: DX[:,i::2,j::2,:] += DY*0.5
    return DX
def update(self,lr): pass
def dump(self): pass
# -------------------------
# Convolution layer

# -------------------------
class Convolution:
  
  def __init__(self,name,write=False):
    
    if write:
      wshape = map(int,list(name.split("-")[-1].split("x")))
      w,h,m,n = wshape
      self.W = numpy.random.normal(0,1/(w*h*m)**.5,wshape)
      self.B = numpy.zeros([n])
      self.name = name
      
    else:
      wshape = map(int,list(name.split("-")[-1].split("x")))
      self.W = numpy.loadtxt(name+’-W.txt’).reshape(wshape)
      self.B = numpy.loadtxt(name+’-B.txt’)
      
  def forward(self,X):
    
    self.X = X
    mb,wx,hx,nx = X.shape
    ww,hw,nx,ny = self.W.shape
    wy,hy = wx-ww+1,hx-hw+1
    Y = numpy.zeros([mb,wy,hy,ny],dtype=’float32’)
    for i in range(ww):
      for j in range(hw):
        Y += numpy.dot(X[:,i:i+wy,j:j+hy,:],self.W[i,j,:,:])
    return Y+self.B

  def gradprop(self,DY):
    
    self.DY = DY
    mb,wy,hy,ny = DY.shape
    ww,hw,nx,ny = self.W.shape
    DX = self.X*0
    for i in range(ww):
      for j in range(hw):
        DX[:,i:i+wy,j:j+hy,:] += numpy.dot(DY,self.W[i,j,:,:].T)
    return DX

  def update(self,lr):
    
    mb,wx,hx,nx = self.X.shape
    mb,wy,hy,ny = self.DY.shape
    ww,hw = wx-wy+1,hx-hy+1
    self.DW = self.W*0
    for i in range(ww):

      for j in range(hw):
        x = self.X[:,i:i+wy,j:j+hy,:]
        dy = self.DY
    self.DW[i,j,:,:] += numpy.tensordot(x,dy,axes=([0,1,2],[0,1,2]))
    self.DB = self.DY.sum(axis=(0,1,2))
    self.W -= lr*self.DW
    self.B -= lr*self.DB
    self.B = numpy.minimum(0,self.B)
  def dump(self):
    numpy.savetxt(self.name+’-W.txt’,self.W.flatten(),fmt=’%.3f’)
    numpy.savetxt(self.name+’-B.txt’,self.B.flatten(),fmt=’%.3f’)
