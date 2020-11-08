# 3D Convolutional Neural Network in Keras/Tensorflow
 Classification CNN for 3D MNIST data.

## Prerequisites
- Linux or Windows 
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- env_mnist3d_cnn.yml

## Getting Started
### Branches
- master: standard implementation of the CNN
- DataGenerator3D: implementation of the CNN using a custom data generator and data augmentation.

### Installation
- Clone or download this repo
- Install dependencies (see env_mnist3d_cnn.yml) and set up your environment

### Dataset
The dataset consists of 12 000 samples of a 3D version of the 2D MNIST dataset. Therefore, 3D point clouds were generated from the images of the digits in the original 2D MNIST dataset. The 3D samples contain 16x16x16 pixels, resulting in 4096 voxels.  

The images are stored as npy-files. The dataset also contains a csv-file with the ID and the corresponding ground truth label.

Download the dataset from: https://www.kaggle.com/daavoo/3d-mnist/data

folder/
- main.py
- DataGenerator.py
- data/
	- sample-0.npy
	- ...
	- labels.csv

where labels.csv contains for instance:

ID; Label \
sample-0; 2 \
sample-1; 7 \
...

### Train and test
Set data directory and define hyperparameters, e.g.:

```
- data_dir = 'data/'
- num_epochs = 50
- batch_size = 32
- train_ratio = 0.7
- validation_ratio = 0.15
- test_ratio = 0.15
```

Run:
```
python main.py
```

## Acknowledgments
- [1] The organization of the dataset is based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
