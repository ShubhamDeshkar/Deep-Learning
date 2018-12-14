from tensorflow import keras
import numpy as np
from sklearn import preprocessing

# calling for MNIST data set from Keras library
MNIST = keras.datasets.mnist

# unpacking the MNIST data set into the following tuples
(tr_img, train_labels), (test_img, test_labels) = MNIST.load_data()

# converting into float values for easier pre-processing later and also
# to suppress the warning from conversion functions that were used later
# in this code
(tr_img, test_img) = (tr_img / 1.0, test_img / 1.0)

# padding the images by 4 by 4 rows and columns of zeros to make it a size of
# 32 by 32 so the features appears in the centre of the feature extractors
train_images = []
for i in range(len(tr_img)):
    print('padding & scaling train-image: ', i+1)
    # does actual padding here
    padded_image = np.pad(tr_img[i], ((2, 2), (2, 2)), 'constant')
    # scales pixel values to range [-0.1, 1.175]
    norm_image = preprocessing.minmax_scale(padded_image, (-0.1, 1.175), copy=False)
    # appends to the train_images list
    train_images.append(norm_image)
# conversion to numpy.ndarray because tensorflow accepts it in this format only
train_images = np.asarray(train_images, dtype=float)

# same as above for test images
test_images = []
for i in range(len(test_img)):
    print('padding & scaling test-image: ', i+1)
    padded_image = np.pad(test_img[i], ((2, 2), (2, 2)), 'constant')
    norm_image = preprocessing.minmax_scale(padded_image, (-0.1, 1.175), copy=False)
    test_images.append(norm_image)
test_images = np.asarray(test_images, dtype=float)

# gathering shape information from the train_images ndarray
train_total = np.shape(train_images)[0]
train_rows = np.shape(train_images)[1]
train_cols = np.shape(train_images)[2]
# this train_images needs to be reshaped to (60000, 32, 32, 1) in order to
# feed to the first layer of the network
train_images = np.reshape(train_images, (train_total, train_rows, train_cols, 1))

# same operations are carried out with test_images
test_total = np.shape(test_images)[0]
test_rows = np.shape(test_images)[1]
test_cols = np.shape(test_images)[2]
test_images = np.reshape(test_images, [test_total, test_rows, test_cols, 1])

# now a variable which is fed to C1 input layer needs to be set according to the
# documentation and keras backend working
input_shape = (train_rows, train_cols, 1)

# converting train_labels and test_labels to one-hot coded vectors
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
# at this point the pre-processing of training images and testing images is done!
#######################################################################################

# 1st layer as convolution, with 6 filters, size (5, 5) and sigmoid activation
C1 = keras.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         strides=(1, 1),
                         padding='valid',
                         data_format='channels_last',
                         activation=keras.activations.relu,
                         use_bias=True,
                         input_shape=input_shape,
                         name='C1'
                         )

# 2nd layer is the sub-sampling layer which down-samples the size of feature map
# by average pooling with pool size 2X2
S2 = keras.layers.AveragePooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='valid',
                                   data_format='channels_last',
                                   name='S2'
                                   )

# 3rd layer is a convolution layer with 16 filters now of the same size as before
# with sigmoid activation
C3 = keras.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         strides=(1, 1),
                         padding='valid',
                         data_format='channels_last',
                         activation=keras.activations.relu,
                         use_bias=True,
                         name='C3'
                         )

# 4th layer is sub-sampling again by average pooling and pool size of 2X2
S4 = keras.layers.AveragePooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='valid',
                                   data_format='channels_last',
                                   name='S4'
                                   )

# 5th layer is the convolution layer with 120 filters with kernel size (5, 5)
# and sigmoid activation
C5 = keras.layers.Conv2D(filters=120,
                         kernel_size=(5, 5),
                         strides=(1, 1),
                         padding='valid',
                         data_format='channels_last',
                         activation=keras.activations.relu,
                         use_bias=True,
                         name='C5'
                         )

# this is a stack to line conversion layer needed to flatten out the output from
# the previous layer in order to feed it to the fully connected layer next
Flatten = keras.layers.Flatten(data_format='channels_last',
                               name='Flatten')

# this is a fully connected layer with 84 units and sigmoid activation
F6 = keras.layers.Dense(units=84,
                        activation=keras.activations.relu,
                        use_bias=True,
                        input_shape=(120,),
                        name='F6'
                        )

# this is a final fully connected layer which is the output layer
# with units = no. of classes and hence softmax activation
F7 = keras.layers.Dense(units=10,
                        activation=keras.activations.softmax,
                        use_bias=True,
                        name='F7'
                        )

# creating a sequential model, where one layer is added after another
MyNet = keras.models.Sequential()

# adding one layer after another sequentially
MyNet.add(C1)
MyNet.add(S2)
MyNet.add(C3)
MyNet.add(S4)
MyNet.add(C5)
MyNet.add(Flatten)
MyNet.add(F6)
MyNet.add(F7)

# compiling the model using optimizer and loss function
MyNet.compile(optimizer=keras.optimizers.SGD(lr=0.1),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.categorical_accuracy]
              )

# training the model using training images and labels
MyNet.fit(train_images, train_labels,
          batch_size=64,
          epochs=13,
          verbose=1,
          shuffle=True
          )

# evaluating the model using testing data
# this achieves a mean squared error of 0.005 and accuracy of 0.9678 at least
test_loss, test_acc = MyNet.evaluate(test_images, test_labels,
                                     batch_size=64,
                                     verbose=1,
                                     )
print(test_loss, test_acc)
