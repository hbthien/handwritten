
#Loading data
#60K training, 10K testing, each of 28x28 pixels = array[size of 784]
import mxnet as mx
mnist = mx.test_utils.get_mnist()

# print(mnist)

#Here we configure the data iterator to feed examples in batches of 100. 
#Keep in mind that each example is a 28x28 grayscale image and the corresponding label.
#Normally, batch is represented by (batch_size, num_channels, width, height)
#here num_channels=1 since only 1 color, w=28, h=28

batch_size = 100
#  initializes the data iterators for the MNIST dataset (train+validation)
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


############### 1.Training with MLP ##################################
#we flatten our 28x28 images into a flat 1-D structure of 784 (28 * 28) raw pixel values. 
#The order of pixel values in the flattened vector does not matter as long as 
#we are being consistent about how we do this across all images

data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

#We declare two fully connected layers with 128 and 64 neurons each. 
#Furthermore, these FC layers are sandwiched between ReLU activation layers 
# each one responsible for performing an element-wise ReLU transformation on the FC layer output.
 
# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax') 
#a loss function computes the cross entropy between the probability distribution (softmax output) 
#predicted by the network and the true probability distribution given by the label.


import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
print("preparing fitting.....")
#this takes the most time of training
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes


#computes the prediction probability scores for each test image. 
#prob[i][j] is the probability that the i-th test image contains the j-th output class.
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
print("preparing predicting.....")
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)


#Since the dataset also has labels for all test images, we can compute the accuracy metric as follows:
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
print("preparing scoring the test data....")
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96



############ 2. Traing with CNN ##############################
print("--------- Training with CNN -----------------")


# defines a CNN architecture called LeNet
data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')



# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)


## Prediction

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = lenet_model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98


