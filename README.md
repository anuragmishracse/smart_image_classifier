# smart_image_classifier 
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/anuragmishracse/smart_image_classifier/blob/master/LICENSE)

**SM**art **I**mage **C**lassifier (**SMIC** hereafter) is a _deep learning_ library built on top of keras using the TensorFlow backend for building models for _image classification_. 
Specialities of this library:
1. It searches for the optimum set of hyperparameters for the classification model
2. Works for any training set, given it's organized in a format that the library understands
3. One can build an Image Classifier in under 5 lines of code

_It is advised that you use a GPU for training your models, as it might take days using a CPU._

---------------------

## Installation
[WIP]

## Requirements
Current implementation of the library depends on the following:
```
1. tqdm
2. pandas 
3. numpy
4. opencv-python
5. tensorflow
6. h5py
7. keras==2.0.9
```
Requirements can be installed by `pip install -r requirements.py`

## Train / Test data organization
The train and test images should be put in seperate directories. The required data format is,
```
/path/to/data/folder/
|---->train
|----|----->trainImage1 #Image names can be anything
|----|----->trainImage2
|----|----->trainImage3
|----|----->___so on___

|---->test
|----|----->testImage1 #Image names can be anything
|----|----->testImage2
|----|----->testImage3
|----|----->___so on___

|---->trainLabels.csv #Contains records in `"trainImage1","cat"` format

|---->testLabels.csv  #Contains records in `"testImage1","dog"` formmat
```

## Usage
Building an image classification model is made really easy. 

```python
from smic import SMIC
clf = SMIC()
clf.prepare_train_data('/path/to/data/folder')
hyperparameters = clf.search_optimal_parameters()
clf.fit(hyperparameters, epochs = 50, batch_size=32)
```

`hyperparameters` is a dict returned by `search_optimal_parameters()` and contains the hyperparameters that seem to work best for the current task at hand. 

If you want you can use your own hyperparameters; skip line [4] and create your own hyperparameters dict like
```python
hyperparameters = {'transfer_model' : 'vgg16', 'optimizer' : 'sgd', 
					'top_layers' : [['dense', 512, 'relu'],['dense', 512, 'relu']]}
```
Pass this dict as an argument to `.fit()`. 

### Supported hyperparameters and values:
```
'transfer_model' : ['vgg16', 'vgg19', 'resnet50', 'inception_v3']
'optimizer': ['sgd', 'rmsprop', 'adam']
'top_layers': A list of all the layers that you want to add on top of the pre-trained CNN.
		Eg: [['dense', 512, 'relu'],['dense', 512, 'relu'],...]
		Here 'dense' is the type of layer, 512 is the output dimension and
		'relu' is the activation function.
```

## TODO
1. An assumption used is that the dataset fits into memory; use batch processing to fit > RAM sized datasets.
2. The hyperparameter tuning currently searches for an optimizer, tranfer learning CNN and number of top layers; Add support for more hyperparameters like momentum value, Dropouts, Regularization etc.
3. Add image data augmentation, that can potentially help learn from fewer datasets.
4. The dataset needs to be organized in the above mentioned format; add support for other formats like:
```
Train
|--->Cat
|--->|----catImage1
|--->|----catImage2
|--->|----......
|--->Dog
|--->|----dogImage1
|--->|----dogImage2
|--->|----......
.............
```
5. Add tests; figure out a way to test changes so that it doesn't corrupt the repo; use CI.


## Note to community
1. A lot of effort needs to be put in, as a community, to develop a systematic approach for hyperparameter tuning, hence suggestions/ ideas welcome.
2. Pull requests are welcome for the above TODO or any other improvement.
3. In case of any issues/ queries, open a new issue or contact me over email.

