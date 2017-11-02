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

	def __init__(data_location, color = 0, dim = 256, multi_gpu='No'):
		self.data_location = data_location
		self.color = color
		self.dim = dim
		self.multi_gpu = multi_gpu
		self.TRAIN_PATH = 'train/'
		self.TEST_PATH = 'test/'
		self.model = None

	def read_image(path):
		image = cv2.imread(path, self.color)
		image = cv2.resize(image, (self.dim, self.dim))
		return image

	def prepare_train_data():
		train = pd.read_csv(os.path.join(self.data_location,'trainImages.csv'))
		train_images = []
		for image_name in tqdm(train['image_id'].values):
		    train_img.append(read_img(TRAIN_PATH + img_path + '.png'))

	def prepare_test_data():
		test = pd.read_csv(os.path.join(self.data_location,'testImages.csv'))

	def search_optimal_hyperparameters():

	def fit(hyperparameters, fine_tune = 'No'):

	def predict(image):
		model.predict(np.asarray([image]))