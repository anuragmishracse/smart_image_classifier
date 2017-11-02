from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, multi_gpu_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

import pandas as pd
import numpy as np
import cv2, os, sys
from tqdm import tqdm

class SMIC():

	def __init__(color = 0, dim = 256, gpu=0):
		self.color = color
		self.dim = dim
		self.gpu = gpu
		self.TRAIN_PATH = 'train/'
		self.TEST_PATH = 'test/'
		self.model = None
		self.label_map = {}
		self.rev_label_map = {}
		self.train_images = []
		self.train_labels = []
		self.hyperparameters = {}

	def read_image(path):
		image = cv2.imread(path, self.color)
		image = cv2.resize(image, (self.dim, self.dim))
		iamge = np.array(image, np.float32) / 255.
		return image

	def prepare_train_data(data_location):
		try:
			train = pd.read_csv(os.path.join(data_location,'trainImages.csv'))
		except:
			print("Error: Invalid location/ File not present.")
		for image_name in tqdm(train['image_id'].values):
			try:
		    	self.train_images.append(self.read_image(os.path.join(data_location, self.TRAIN_PATH, image_name)))
		    except Exception as e:
		    	print("Error reading image: " + repr(e))
		    	print("Breaking.")
		    	break

		labels = train['label'].tolist()
		self.label_map = {k:v for v,k in enumerate(set(labels))}
		self.rev_label_map = {v:k for v,k in enumerate(set(labels))}
		self.train_labels = np.asarray([self.label_map[label] for label in labels])

	def prepare_test_data():
		pass

	def search_optimal_hyperparameters(samples_per_class = 50, check_epochs = 10):
		labels_catgorical = to_categorical(self.train_labels)
		return hyperparameters

	# hyperparameters = {
	# 'transfer_model':'resnet50',
	# 'top_layers':[(),()],
	# 'optimization_algo':'SGD',
	# }

	def fit(hyperparameters, epochs, batch_size, fine_tune = False):
		labels_catgorical = to_categorical(self.train_labels)
		model = create_model(hyperparameters)
		if self.gpu > 1:
			model = multi_gpu_model(model, self.gpu)
		model = compile_model(hyperparameters)
		model.fit(np.asarray(self.train_images), np.asarray(labels_catgorical), batch_size=batch_size, epochs = epochs, validation_split = 0.1)

		if fine_tune:
			for layer in model.layers:
				layer.trainable = False
			model.compile(loss='categorical_crossentropy', optimizer= optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
			model.fit(np.asarray(self.train_images), np.asarray(labels_catgorical), batch_size=batch_size, epochs = epochs, validation_split = 0.1)			
		return model

	def predict(image):
		prediction = model.predict(np.asarray([image]))
		prediction = np.argmax(prediction, axis=1)
		return self.rev_label_map[prediction]

	def visualize(summary=False):
		if summary:
			print self.model.summary()