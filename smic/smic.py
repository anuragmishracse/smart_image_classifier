from keras import applications
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, multi_gpu_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

import pandas as pd
import numpy as np
import cv2, os, sys, ast
from tqdm import tqdm
from collections import defaultdict

class SMIC():

	def __init__(self, color = 1, dim = 256, gpu=0):
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
		self.num_classes = -1
		self.hyperparameters = {}
		self.transfer_models = {'vgg16' : VGG16, 'vgg19' : VGG19, 'resnet50' : ResNet50, 'inception_v3' : InceptionV3}
		self.optimizers = {'sgd' : 'SGD', 'rmsprop' : 'RMSprop', 'adam' : 'Adam'}
		self.layers = {'dense' : Dense, 'dropout' : Dropout}

	def read_image(self, path):
		image = cv2.imread(path, self.color)
		image = cv2.resize(image, (self.dim, self.dim))
		iamge = np.array(image, np.float32) / 255.
		return image

	def prepare_train_data(self, data_location):
		try:
			train = pd.read_csv(os.path.join(data_location,'trainLabels.csv'))
			for image_name in tqdm(train['image_id'].values):
				try:
					self.train_images.append(self.read_image(os.path.join(data_location, self.TRAIN_PATH, image_name)+'.png'))
				except Exception as e:
					print("Error reading image: " + repr(e))
		except:
			print("Error: Invalid location/ File not present.")
			exit()

		labels = train['label'].tolist()
		self.num_classes = len(set(labels))
		self.label_map = {k:v for v,k in enumerate(set(labels))}
		self.rev_label_map = {v:k for v,k in enumerate(set(labels))}
		self.train_labels = np.asarray([self.label_map[label] for label in labels])

	def prepare_test_data(self):
		pass

	def create_model(self, hyperparameters):
		base_model = self.transfer_models[hyperparameters['transfer_model']](weights='imagenet', include_top=False, input_shape=(self.dim, self.dim, 3))
		for layer in base_model.layers:
			layer.trainable=False
		classifier = Flatten()(base_model.output)
		for layer_param in hyperparameters['top_layers']:
			classifier = self.layers[layer_param[0]](layer_param[1], activation=layer_param[2])(classifier)
		classifier = Dense(self.num_classes, activation='softmax')(classifier)
		model = Model(base_model.input, classifier)

		model.compile(loss='categorical_crossentropy', optimizer = self.optimizers[hyperparameters['optimizer']], metrics=['accuracy'])
		return model

	def search_optimal_hyperparameters(self, samples_per_class = 50, check_epochs = 10):
		search_result = {}

		sample_train_images=[]
		sample_labels_catgorical=[]

		class_image_dict = defaultdict(list)
		for ind in range(len(self.train_labels)):
			class_image_dict[self.train_labels[ind]].append(self.train_images[ind])

		for class_name in class_image_dict.keys():
			class_images = class_image_dict[class_name][:samples_per_class]
			sample_train_images.extend(class_images)
			sample_labels_catgorical.extend([class_name]*len(class_images))

		sample_labels_catgorical = to_categorical(sample_labels_catgorical)

		for transfer_model in self.transfer_models.keys():
			for optimizer in self.optimizers.keys():
				layers=[]
				for layer_count in range(1,2):
					layers.append(['dense', 512, 'relu']) 
					hyperparameters={'transfer_model' : transfer_model, 'optimizer' : optimizer, 'top_layers' : layers}
					model = self.create_model(hyperparameters)
					history = model.fit(np.asarray(sample_train_images), np.asarray(sample_labels_catgorical), batch_size=32, epochs = check_epochs, validation_split = 0.1)
					print history.history['acc']
					search_result[str(hyperparameters)]=[history.history['acc'][-1], history.history['val_acc'][-1]]

		print search_result

		for hyperparameters, results in search_result.items():
			if results[1] > results[0]*1.05:
				del search_result[hyperparameters]

		search_result = sorted(search_result.items(), key = lambda x: x[1], reverse=True)
		print search_result
		return ast.literal_eval(search_result[0][0])


	def fit(self, hyperparameters, epochs, batch_size, fine_tune = False):
		labels_categorical = to_categorical(self.train_labels)
		self.model = self.create_model(hyperparameters)
		history = self.model.fit(np.asarray(self.train_images), np.asarray(labels_categorical), batch_size=batch_size, epochs = epochs, validation_split = 0.1)

		if fine_tune:
			for layer in self.model.layers:
				layer.trainable = False
			self.model.compile(loss='categorical_crossentropy', optimizer= SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
			history_fine = self.model.fit(np.asarray(self.train_images), np.asarray(labels_categorical), batch_size=batch_size, epochs = epochs, validation_split = 0.1)			
			history.extend(history_fine)
		return history

	def predict(self, image):
		prediction = self.model.predict(np.asarray([image]))
		prediction = np.argmax(prediction, axis=1)
		return self.rev_label_map[prediction]

	def visualize(self, summary=False):
		if summary:
			print self.model.summary()

	def save(self, path):
		self.model.save(path)

	def load(self, path):
		self.model = load_model(path)
