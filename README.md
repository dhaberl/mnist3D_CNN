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

Download the dataset from: [LINK] 

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

### Dataset
A subset of 42 000 grey-scale images of the original MNIST database was used. Each image contains 28x28 pixels, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the brightness (low values) or darkness (high values) of that pixel. This pixel-value is an integer between 0 (white) and 255 (black). 

The images are stored as npy-files. The dataset also contains a csv-file with the ID and the corresponding ground truth label.

Download the dataset from: [LINK] 

folder/
- main.py
- DataGenerator.py
- data/
	- img_0.npy
	- ...
	- labels.csv

where labels.csv contains for instance:

ID; Label \
img_0; 2 \
img_1; 7 \
...

### Data Generator
The Data Generator generates the dataset in batches on multiple cores for real-time data feeding to the machine learning model. 

The generator can be used by importing it in the main file:

```
from DataGenerator import DataGenerator
```

Input parameters are:

- data_dir: path to the data directory (string)
- list_ids: list of IDs as shown above (list)
- labels: list of labels as shown above (list)
- batch_size: number of samples that will be propagated through the network (integer)
- dim: dimensions of the data (tuple with intergers). E.g., image with 16x16x16 voxels => (16, 16, 16)
- n_channels: number of channels (integer). E.g., RGB = 3 channels
- n_classes: number of classes (integer)
- shuffle: whether to shuffle at generation or not (boolean) 
- rescale: whether to rescale the samples before generation or not (boolean).
- output_dim: dimensions of the samples after rescaling (tuple with integers). E.g., rescaling a sample with original dim=(16, 16, 16) to output_dim=(28, 28, 28) 
- **da_parameters

### Data augmentation

The Data Generator also allows real-time data augmentation.

For example:

```
da_parameters = {"width_shift": 5.,
                 "height_shift": 5.,
                 "depth_shift": 5.,
                 "rotation_range": 15.,
                 "horizontal_flip": 0.5,
                 "vertical_flip": 0.5,
                 "min_zoom": 0.7,
                 "max_zoom": 1.1,
                 "random_crop_size": 0.85,
                 "random_crop_rate": 1.,
                 "center_crop_size": 0.85,
                 "center_crop_rate": 1.,
                 "gaussian_filter_std": 1.,
                 "gaussian_filter_rate": 1.
                 }
```

### Train and test
Set data directory and define hyperparameters, e.g.:

```
- data_dir = 'data/'
- rescale = False
- output_dim = (28, 28, 28) # only needed when rescale is True
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
