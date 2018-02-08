import sys; print('Python %s on %s' % (sys.version, sys.platform))

# make sure to connect via 'xterm -ls -xrm 'XTerm*selectToClipboard: true'&
import os
# set the environment variable
os.environ["KERAS_BACKEND"] = "tensorflow"
# make sure we were successful
print(os.environ.get("KERAS_BACKEND"))

# set preloading
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.metrics import confusion_matrix
from os.path import expanduser

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima dataset
nr_epochs = 200

url = 'https://github.com/RaikOtto/DeepLearningTutorial/raw/master/pima-indians-diabetes.csv'
response = urllib.request.urlopen( url )
dataset = numpy.loadtxt(response, delimiter=",")
    
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Fit the model
history = model.fit(
    X,
    Y,
    epochs=nr_epochs,
    batch_size=10,
    verbose=2)

# Evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# calculate predictions
probabilities = model.predict(X)
predictions = [float(x>0.5) for x in probabilities]

## visualizations WORK ONLY VIA 'ssh -XC' and 'xterm python' or locally on a laptop

# loss function
import gnuplotlib as gpl

loss = history.history['loss']

epochs = range(1, len(loss) + 1)

gpl.plot(epochs, loss,
	_with='points', 
	title='Training loss', 
	legend='Training loss', 
	xlabel='Epochs',
	ylabel='Loss',
	unset='grid',
	terminal='dumb 40,80')

# accuracy

plt.clf()   # clear figure

acc = history.history['acc']

gpl.plot(epochs, acc,
	_with='points', 
	title='Training accuracy', 
	legend='Training acc', 
	xlabel='Epochs',
	ylabel='Accuracy',
	unset='grid',
	terminal='dumb 40,80')

#

cm = confusion_matrix(Y, predictions)
tp = float(cm[1,1])
fp = float(cm[0,1])
tn = float(cm[0,0])
fn = float(cm[1,0])
print ("True positives:  %.0f" % tp)
print ("False positives: %.0f" % fp)
print ("True negatives:  %.0f" % tn)
print ("False negatives: %.0f" % fn)

prec = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = (2*prec*rec)/(prec+rec)
print ("Precision: %.3f" % prec)
print ("Recall: %.3f" % rec)
print ("F1: %.3f" % f1)