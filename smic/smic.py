from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

import pandas as pd
import numpy as np
import cv2, os, sys
from tqdm import tqdm

class SMIC():

	def __init__(color = 0, dim = 256, multi_gpu='No'):
		self.color = color
		self.dim = dim
		self.multi_gpu = multi_gpu
		self.TRAIN_PATH = 'train/'
		self.TEST_PATH = 'test/'
		self.model = None
		self.train_images = []

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
		    	print repr(e)

	def prepare_test_data():
		pass

	def search_optimal_hyperparameters():
		pass

	def fit(hyperparameters, fine_tune = 'No'):
		pass
		
	def predict(image):
		model.predict(np.asarray([image]))



