# smart_image_classifier 
## [Work in Progress][Expected Release: 07/11/2017]
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/anuragmishracse/smart_image_classifier/blob/master/LICENSE)

**SM**art **I**mage **C**lassifier (**SMIC** hereafter) is a _deep learning_ library built on top of keras using the TensorFlow backend for building models for _image classification_. 
Specialities of this library:
1. It searches for the optimum set of hyperparameters for the classification model
2. Works for any training set, given it's organized in a format that the library understands
3. One can build an Image Classifier in under 5 lines of code

_It is advised that you use a GPU for training your models, it would take days using a CPU._

---------------------

## Installation
[TODO]

## Requirements
Current implementation of the library depends on the following:
```
1. tqdm
2. pandas 
3. numpy
4. cv2
5. tensorflow
6. h5py
7. keras==2.0.9
```
Requirements can be installed by `pip install -r requirements.py`

## Train / Test data organization
[TODO]

## Usage
Building an image classification model is made really easy. 

```python
from smic import SMIC
clf = SMIC()
clf.prepare_train_data('/path/to/image/data')
hp = clf.search_optimal_parameters()
clf.fit(hp, epochs = 50, batch_size=32)
```

## TODO
[TODO]

## Note to community
1. A lot of effort needs to be put in, as a community, to develop a systematic approach for hyperparameter tuning, hence suggestions/ ideas welcome.
2. Pull requests are welcome for the above TODO or any other improvement.
3. In case of any issue, open a new issue or contact me over email.

## References
[TO ADD]
