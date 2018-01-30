# Link to the tutorial: http://heatmapping.org/wifs2017/training.pdf

# YOU CAN GET THE REQUIRED DATA HERE
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

import numpy,scipy,scipy.io
Xtrain = []
Ttrain = []
for batch in [1,2,3,4,5]:
  D = scipy.io.loadmat(’cifar-10-batches-mat/data_batch_%d’%batch)
  Xtrain += [(D[’data’]/127.5-1.0).reshape([-1,3,32,32]).transpose([0,2,3,1])]
  Ttrain += [(D[’labels’][:,numpy.newaxis] == numpy.arange(10)).reshape([-1,1,1,10])*1.0]
  Xtrain = numpy.concatenate(Xtrain,axis=0)
  Ttrain = numpy.concatenate(Ttrain,axis=0)


D = scipy.io.loadmat(’cifar-10-batches-mat/test_batch’)
Xtest = (D[’data’][:500]/127.5-1.0).reshape([-1,3,32,32]).transpose([0,2,3,1])
Ttest = (D[’labels’][:500][:,numpy.newaxis] == numpy.arange(10)).reshape([-1,1,1,10])

%matplotlib inline
import utils
utils.images(Xtest[:50])

nn = md.Network([
md.Convolution(’cnn/c1-5x5x3x20’,write=True),md.ReLU(),md.Pooling(),
md.Convolution(’cnn/c2-5x5x20x50’,write=True),md.ReLU(),md.Pooling(),
md.Convolution(’cnn/c3-4x4x50x200’,write=True),md.ReLU(),md.Pooling(),
md.Convolution(’cnn/c4-1x1x200x10’,write=True),
])
nn.layers[-1].W*=0

def trainingstep():
  # 1. select random mini-batch
  R = numpy.random.permutation(len(Xtrain))[:25]
  x,t = Xtrain[R],Ttrain[R]
  # 2. forward pass
  y = nn.forward(x)
  # 3. error gradient
  dy = numpy.exp(y) / (numpy.exp(y).sum(axis=1)[:,numpy.newaxis] + 1) - t
  # 4. backward pass
  nn.gradprop(dy)
  2
  # 5. updating the parameters with some learning rate
  nn.update(0.001)

def teststep(i):
  # 1. store the network
  nn.dump()
  # 2. compute prediction labels
  pred=numpy.argmax(nn.forward(Xtest),axis=-1)
  # 3. extract target labels
  targ=numpy.argmax(Ttest,axis=-1)
  # 4. check how often they match (this gives a measure of predictive accuracy)
  print(’%6d %.3f’%(i,(pred==targ).mean()))

import numpy,os
cpt = [10,20,40,100,200,400,1000,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000]
for i in range(1,20001):
  trainingstep()
if i in cpt: teststep(i)
