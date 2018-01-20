# edit from original resource: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import *
from imutils import paths
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("dataset"))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	drive, path_and_file = os.path.splitdrive(imagePath)
	path, file = os.path.split(path_and_file)
	label = file.split(".")[0]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(label)

	# show an update every 24 images
	if i > 0 and i % 24 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 24)

# define the architecture of the network
model = Sequential()
model.add(Dense(290, input_dim=3072, kernel_initializer ='uniform', activation='relu'))
model.add(Dense(145, kernel_initializer ='uniform', activation='relu'))
model.add(Dense(24))
model.add(Activation("softmax"))

# train the model using Adam
print("[INFO] compiling model...")
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.fit(data, labels, epochs=50, batch_size=128, verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss0, accuracy0) = model.evaluate(data, labels,batch_size=128, verbose=1)
print("[INFO] data train loss={:.4f}, accuracy: {:.4f}%".format(loss0,accuracy0 * 100))

# save model
model.save('model.h5')
print("Saved model to disk")