import os
import keras
import numpy as np
from scipy import ndimage
import random


class DataGenerator(keras.utils.Sequence):
    """Generates data for keras"""
    def __init__(self, data_dir, list_ids, labels, batch_size=100, dim=(16, 16, 16), n_channels=1, n_classes=10,
                 shuffle=True, rescale=False, output_dim=(28, 28, 28), width_shift=0.0, height_shift=0.0,
                 depth_shift=0.0, rotation_range=0.0, horizontal_flip=0.0, vertical_flip=0.0, min_zoom=0.0,
                 max_zoom=0.0, random_crop_size=0.0, random_crop_rate=0.0, center_crop_size=0.0, center_crop_rate=0.0,
                 gaussian_filter_std=0.0, gaussian_filter_rate=0.0):
        """Initizilation"""
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.classes = []
        self.shuffle = shuffle
        self.rescale = rescale
        self.output_dim = output_dim
        self.width_shift = width_shift
        self.height_shift = height_shift
        self.depth_shift = depth_shift
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.random_crop_size = random_crop_size
        self.random_crop_rate = random_crop_rate
        self.center_crop_size = center_crop_size
        self.center_crop_rate = center_crop_rate
        self.gaussian_filter_std = gaussian_filter_std
        self.gaussian_filter_rate = gaussian_filter_rate
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoche"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        # Add true labels to classes attribute
        self.classes += y.tolist()

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __Rescale(self, sample_temp):
        if self.rescale is True:
            r = self.dim[0]/self.output_dim[0]
            new_sample_temp = np.zeros((*self.output_dim, self.n_channels))

            height, width, depth, intensity = new_sample_temp.shape
            zheight = int(np.round(r*height))
            zwidth = int(np.round(r*width))
            zdepth = int(np.round(r*depth))
            zintensity = intensity

            y = (height - zheight) // 2
            x = (width - zwidth) // 2
            z = (depth - zdepth) // 2
            layer = (intensity - zintensity) // 2
            new_sample_temp[y:y + zheight, x:x + zwidth, z:z + zdepth, layer:layer + zintensity] = \
                sample_temp[0:zheight, 0:zwidth, 0:zdepth, 0:zintensity]

            return new_sample_temp
        else:
            return sample_temp

    def __Translation(self, sample_temp):
        rand_x = random.uniform(-self.width_shift, self.width_shift)
        sample_temp = ndimage.interpolation.shift(sample_temp, (0, rand_x, 0, 0), order=5, mode='constant')

        rand_y = random.uniform(-self.height_shift, self.height_shift)
        sample_temp = ndimage.interpolation.shift(sample_temp, (rand_y, 0, 0, 0), order=5, mode='constant')

        rand_z = random.uniform(-self.depth_shift, self.depth_shift)
        sample_temp = ndimage.interpolation.shift(sample_temp, (0, 0, rand_z, 0), order=5, mode='constant')

        return sample_temp

    def __Rotation(self, sample_temp):
        theta = random.uniform(-self.rotation_range, self.rotation_range)
        sample_temp = ndimage.interpolation.rotate(sample_temp, theta, reshape=False, order=5, mode='constant')

        return sample_temp

    def __Flip(self, sample_temp):
        # horizontal
        if self.horizontal_flip != 0:
            rand_hflip = random.random()
            if rand_hflip <= self.horizontal_flip:
                sample_temp = np.fliplr(sample_temp)
                return sample_temp
            else:
                return sample_temp
        # vertical
        if self.vertical_flip != 0:
            rand_vflip = random.random()
            if rand_vflip <= self.vertical_flip:
                sample_temp = np.flipud(sample_temp)
                return sample_temp
            else:
                return sample_temp
        else:
            return sample_temp

    def __Zoom(self, sample_temp):
        if self.min_zoom != 0 and self.max_zoom != 0:
            r = random.uniform(self.min_zoom, self.max_zoom)
            height, width, depth, intensity = sample_temp.shape
            zheight = int(np.round(r*height))
            zwidth = int(np.round(r*width))
            zdepth = int(np.round(r*depth))
            zintensity = intensity

            if r < 1.0:
                new_sample_temp = np.zeros_like(sample_temp)
                y = (height - zheight) // 2
                x = (width - zwidth) // 2
                z = (depth - zdepth) // 2
                layer = (intensity-zintensity) // 2
                new_sample_temp[y:y + zheight, x:x + zwidth, z:z + zdepth, layer:layer + zintensity] = \
                    ndimage.interpolation.zoom(sample_temp, (float(r), float(r), float(r), 1.0), order=5,
                                               mode='constant')[0:zheight, 0:zwidth, 0:zdepth, 0:zintensity]
                return new_sample_temp

            elif r > 1.0:
                y = (zheight - height) // 2
                x = (zwidth - width) // 2
                z = (zdepth - depth) // 2
                layer = (zintensity - intensity) // 2
                new_sample_temp = ndimage.interpolation.zoom(sample_temp[y:y + zheight,
                                                             x:x + zwidth, z:z + zdepth, layer:layer + zintensity],
                                                             (float(r), float(r), float(r), 1.0), order=5,
                                                             mode='constant')

                extrah = (new_sample_temp.shape[0] - height) // 2
                extraw = (new_sample_temp.shape[1] - width) // 2
                extrad = (new_sample_temp.shape[2] - depth) // 2
                extrai = (new_sample_temp.shape[3] - intensity) // 2
                new_sample_temp = \
                    new_sample_temp[extrah:extrah + height,
                                    extraw:extraw + width,
                                    extrad:extrad + depth,
                                    extrai:extrai + intensity]
                return new_sample_temp

            else:
                return sample_temp
        else:
            return sample_temp

    def __Random_Crop(self, sample_temp):
        if self.random_crop_size != 0 and self.random_crop_rate != 0:
            r = random.random()
            height, width, depth, intensity = sample_temp.shape
            dy = int(np.round(self.random_crop_size*height))
            dx = int(np.round(self.random_crop_size*width))
            dz = int(np.round(self.random_crop_size*depth))

            if r <= self.random_crop_rate:
                x = np.random.randint(0, width - dx + 1)
                y = np.random.randint(0, height - dy + 1)
                z = np.random.randint(0, depth - dz + 1)
                sample_temp = sample_temp[y:(y + dy), x:(x + dx), z:(z + dz), :]
                sample_temp = ndimage.interpolation.zoom(sample_temp,
                                                         (float(height/dy), float(width/dx), float(depth/dz), 1.0),
                                                         order=5, mode='constant')
                return sample_temp
            else:
                return sample_temp
        else:
            return sample_temp

    def __Center_Crop(self, sample_temp):
        if self.center_crop_size != 0 and self.center_crop_rate != 0:
            r = random.random()
            height, width, depth, intensity = sample_temp.shape
            dy = int(np.round(self.center_crop_size*height))
            dx = int(np.round(self.center_crop_size*width))
            dz = int(np.round(self.center_crop_size*depth))

            if r <= self.center_crop_rate:
                x = (height - dx) // 2
                y = (width - dy) // 2
                z = (depth - dz) // 2
                sample_temp = sample_temp[y:(y + dy), x:(x + dx), z:(z + dz), :]
                sample_temp = ndimage.interpolation.zoom(sample_temp,
                                                         (float(height/dy), float(width/dx), float(depth/dz), 1.0),
                                                         order=5, mode='constant')
                return sample_temp
            else:
                return sample_temp
        else:
            return sample_temp

    def _Gaussian_Filter(self, sample_temp):
        if self.gaussian_filter_std != 0 and self.gaussian_filter_rate != 0:
            r = random.random()
            if r <= self.gaussian_filter_rate:
                sample_temp = ndimage.gaussian_filter(sample_temp,
                                                      sigma=random.uniform(0, self.gaussian_filter_std))
                return sample_temp
            else:
                return sample_temp
        else:
            return sample_temp

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # X: (n_samples, *dim, n_channels)
        # Initialization
        if self.rescale is True:
            X = np.empty((self.batch_size, *self.output_dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Load samples
            sample_path = os.path.join(self.data_dir, ID + ".npy")
            sample_temp = np.load(sample_path).astype("float32")

            # Rescale 3D image
            sample_temp = self.__Rescale(sample_temp)

            # Data Augmentation
            random.seed()

            # Combinations
            random_number = random.randint(1, 9)

            # Translation
            if random_number == 1:
                sample_temp = self.__Translation(sample_temp)

            # Rotation
            if random_number == 2:
                sample_temp = self.__Rotation(sample_temp)

            # Flip
            if random_number == 3:
                sample_temp = self.__Flip(sample_temp)

            # Zoom
            if random_number == 4:
                sample_temp = self.__Zoom(sample_temp)

            # Random crop
            if random_number == 5:
                sample_temp = self.__Random_Crop(sample_temp)

            # Center crop
            if random_number == 6:
                sample_temp = self.__Center_Crop(sample_temp)

            # Gaussian Filter
            if random_number == 7:
                sample_temp = self._Gaussian_Filter(sample_temp)

            # Translation + Rotation
            if random_number == 8:
                sample_temp = self.__Translation(sample_temp)
                sample_temp = self.__Rotation(sample_temp)

            # Translation + Rotation + Zoom
            if random_number == 9:
                sample_temp = self.__Translation(sample_temp)
                sample_temp = self.__Rotation(sample_temp)
                sample_temp = self.__Zoom(sample_temp)

            # Store (augmented) sample
            X[i, ] = sample_temp

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
