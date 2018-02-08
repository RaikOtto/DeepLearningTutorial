# A Small Example (Boston Housing Data)
# Building a model in Keras starts by constructing an empty Sequential model.

# Set # of GPUs to be used in TensorFlow backend.
Sys.setenv(CUDA_VISIBLE_DEVICES="0,3")

# Limit the # of threads to keep the memory footprint low.
Sys.setenv(OPENBLAS_NUM_THREADS="1")
Sys.setenv(NUMEXPR_NUM_THREADS="1")
Sys.setenv(OMP_NUM_THREADS="1")

# below commands will throw error don't panic!
library(tensorflow) # only loaded earlier for setting parameters
config = tf$ConfigProto()
result = tryCatch({ config$gpu_options$allow_growth=TRUE }, error = function(e) { })
print(config$gpu_options$allow_growth)
config$inter_op_parallelism_threads = 1L
config$intra_op_parallelism_threads = 1L
server = tf$train$Server$create_local_server(config=config)
sess = tf$Session(server$target)

# load libraries
library(kerasR) # if you get an error message -> install.packages("kerasR")
library(keras)

k_set_session(sess)

nr_epochs = 2 # too low in practice but needed due to performance issues

mod <- Sequential()
# The result of Sequential, as with most of the functions provided by kerasR, is a python.builtin.object. This object type, defined from the reticulate package, provides direct access to all of the methods and attributes exposed by the underlying python class. To access these, we use the $ operator followed by the method name. Layers are added by calling the method add. This function takes as an input another python.builtin.object, generally constructed as the output of another kerasR function. For example, to add a dense layer to our model we do the following:
  
mod$add(Dense(units = 50, input_shape = 13))
# We have now added a dense layer with 200 neurons. The first layer must include a specification of the input_shape, giving the dimensionality of the input data. Here we set the number of input variables equal to 13. Next in the model, we add an activation defined by a rectified linear unit to the model:
  
mod$add(Activation("relu"))
# Now, we add a dense layer with just a single neuron to serve as the output layer:
  
mod$add(Dense(units = 1))

# Once the model is fully defined, we have to compile it before fitting its parameters or using it for prediction. Compiling a model can be done with the method compile, but some optional arguments to it can cause trouble when converting from R types so we provide a custom wrapper keras_compile. At a minimum we need to specify the loss function and the optimizer. The loss can be specified with just a string, but we will pass the output of another kerasR function as the optimizer. Here we use the RMSprop optimizer as it generally gives fairly good performance:
  
keras_compile(mod,  loss = 'mse', optimizer = RMSprop())
# Now we are able to fit the weights in the model from some training data, but we do not yet have any data from which to train! Let’s load some using the wrapper function load_boston_housing. We provide several data loading functions as part of the package, and all return data in the same format. In this case it will be helpful to scale the data matrices:
  
boston <- load_boston_housing()
X_train <- scale(boston$X_train)
Y_train <- boston$Y_train
X_test <- scale(boston$X_test)
Y_test <- boston$Y_test

# Now, we call the wrapper keras_fit in order to fit the model from this data. As with the compilation, there is a direct method for doing this but you will likely run into data type conversion problems calling it directly. Instead, we see how easy it is to use the wrapper function (if you run this yourself, you will see that Keras provides very good verbose output for tracking the fitting of models):
  
keras_fit(mod, X_train, Y_train,
          batch_size = 32, epochs = nr_epochs,
          verbose = 1, validation_split = 0.1)

## output
# Notice that the model does not do particularly well here, probably due to over-fitting on such as small set.

pred <- keras_predict(mod, normalize(X_test))
sd(as.numeric(pred) - Y_test) / sd(Y_test)

## A Larger Example (MNIST)
# To show the power of neural networks we need a larger dataset to make use of. A popular first dataset for applying neural networks is the MNIST Handwriting dataset, consisting of small black and white scans of handwritten numeric digits (0-9). The task is to build a classifier that correctly identifies the numeric value from the scan. We may load this dataset in with the following:
  
mnist <- load_mnist()
X_train <- mnist$X_train
Y_train <- mnist$Y_train
X_test <- mnist$X_test
Y_test <- mnist$Y_test
dim(X_train)

# Notice that the training data shape is three dimensional (in the language of Keras this is a tensor). The first dimension is the specific sample number, the second is the row of the scan, and the third is the column of the scan. We will use this additional spatial information in the next section, but for now let us flatten the data so that is is just a 2D-Tensor. The values are pixel intensities between 0 and 255, so we will also normalize the values to be between 0 and 1:
  
X_train <- array(X_train, dim = c(dim(X_train)[1], prod(dim(X_train)[-1]))) / 255
X_test <- array(X_test, dim = c(dim(X_test)[1], prod(dim(X_test)[-1]))) / 255
# Finally, we want to process the response vector y into a different format as well. By default it is encoded in a one-column matrix with each row giving the number represented by the hand written image. We instead would like this to be converted into a 10-column binary matrix, with exactly one 1 in each row indicating which digit is represented. This is similar to the factor contrasts matrix one would construct when using factors in a linear model. In the neural network literature it is call the one-hot representation. We construct it here via the wrapper function to_categorical. Note that we only want to convert the training data to this format; the test data should remain in its original one-column shape.

Y_train <- to_categorical(mnist$Y_train, 10)
# With the data in hand, we are now ready to construct a neural network. We will create three blocks of identical Dense layers, all having 512 nodes, a leaky rectified linear unit, and drop out. These will be followed on the top output layer of 10 nodes and a final softmax activation. These are fairly well-known choices for a simple dense neural network and allow us to show off many of the possibilities within the kerasR interface:
  
mod <- Sequential()

mod$add(Dense(units = 512, input_shape = dim(X_train)[2]))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = 512))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = 512))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(10))
mod$add(Activation("softmax"))
# We then compile the model with the “categorical_crossentropy” loss and fit it on the training data:
  
keras_compile(mod,  loss = 'categorical_crossentropy', optimizer = RMSprop())
keras_fit(
    mod,
    X_train,
    Y_train,
    batch_size = 32,
    epochs = nr_epochs,
    verbose = 1,
    validation_split = 0.1
)
# Now that the model is trained, we could use the function keras_predict once again, however this would give us an output matrix with 10 columns. It is not too much work to turn this into predicted classes, but kerasR provides keras_predict_classes that extracts the predicted classes directly. Using this we are able to evaluate the data on the test set.

Y_test_hat <- keras_predict_classes(mod, X_test)
table(Y_test, Y_test_hat)
mean(Y_test == Y_test_hat)
##       Y_test_hat
## Y_test         0    1    2    3    4    5    6    7    8    9
##           0  952    1    5    0    1   11    4    1    3    2
##           1    0 1121    5    0    0    1    2    0    6    0
##           2    1    4  987    0   16    0    3    9   12    0
##           3    0    1   17  946    3   13    0   15    6    9
##           4    0    0    3    0  965    0    1    1    1   11
##           5    2    1    2   17    8  812    9    2   27   12
##           6    4    3    2    0   13    7  923    0    6    0
##           7    1    7    8    1    4    1    0  999    0    7
##           8    2    1    6    7    5    4    0    8  937    4
##           9    1    7    1    5   18    3    1   13    2  958
## [1] 0.96
# Looking at the mis-classification rate and the confusion matrix, we see that the neural network performs very well (with a classification rate around 95%). It’s possible to get slightly higher with strictly dense layers by employing additional tricks and using larger models with more regularization. To increase the model drastically requires the use of convolutional neural networks (CNN), which we will look at in the next section.

# Convolutional neural networks
# To begin, we load the MNIST dataset in once again, but this time increase the number of dimension in the X_train tensor by one rather than reducing it by one. These images are black and white and one way to think about this additional dimension is that it represents a “gray” channel.
