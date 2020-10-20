import os
import random
import numpy as np
from PIL import Image, ImageStat
from keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import random_brightness, random_shift, random_rotation, random_zoom

# augmentations
brightness_dec = lambda im : random_brightness(im, (0.2, 1))
brightness_inc = lambda im : random_brightness(im, (1, 3))
shift = lambda im : random_shift(im, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2)
rotation = lambda im : random_rotation(im, 20, row_axis=0, col_axis=1, channel_axis=2)
zoom_in = lambda im : random_zoom(im, (0.85, 0.85), row_axis=0, col_axis=1, channel_axis=2)
zoom_out = lambda im : random_zoom(im, (1.15, 1.15), row_axis=0, col_axis=1, channel_axis=2)


class ImageDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, sample_k, mode, batch_size=32, dim=(70, 70), n_channels=1, n_classes=21, aug=None,
                 shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param sample_k: number of sample instances
        :param mode: train, validation, test
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param aug: list of augmentations
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.sample_k = sample_k
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.aug = aug
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # '/home/toertei/PycharmProjects/traffic-signs/traffic_signs_train/germany/20/00001_00019.ppm'
        if self.mode == 'training':
            X_data, Y_data = np.empty((self.batch_size * self.sample_k * (len(self.aug)+1), *self.dim, self.n_channels)), \
                             np.empty((self.batch_size * self.sample_k * (len(self.aug)+1), self.n_classes))
        elif self.mode == 'validation':
            X_data, Y_data = None, None
        elif self.mode == 'testing':
            X_data = np.empty((self.batch_size, *self.dim, self.n_channels))


        for i, ID in enumerate(list_IDs_temp):
            root_fldr, country, label, sample, extension = self.__parse_datapath(ID)
            samples_k = np.arange(self.sample_k)

            # aleviate problem of cardinality imbalance between training subsets
            if self.mode == 'training':
                if country == 'germany':
                    samples_k = [random.randint(0, 29) for _ in range(self.sample_k)]

                X = np.empty((self.sample_k * (len(self.aug)+1), *self.dim, self.n_channels))
                existing_path = ''
                for j in range(self.sample_k):
                    sample_path = os.path.join(root_fldr, str(label),
                                               f"{sample:05d}_{samples_k[j]:05d}.{extension}")
                    if not os.path.exists(sample_path):
                        sample_path = existing_path
                    else:
                        existing_path = sample_path

                    img, brightness = self.__load_sample(sample_path)
                    X[j*(len(self.aug)+1): (j+1)*(len(self.aug)+1), ] = self.__augmentations(img, country, brightness)

                Y = to_categorical([label] * X.shape[0], self.n_classes)
                X_data[i * self.sample_k * (len(self.aug)+1): (i + 1) * self.sample_k * (len(self.aug)+1), ] = X
                Y_data[i * self.sample_k * (len(self.aug)+1): (i + 1) * self.sample_k * (len(self.aug)+1)] = Y

            # sample 'samples from germany' uniformly
            elif self.mode == 'validation':
                if country == 'germany':
                    samples_k = [x for x in range(0, self.n_classes + 1, self.n_classes // self.sample_k)]

                for j in range(self.sample_k):
                    sample_path = os.path.join(root_fldr, str(label), f"{sample:05d}_{samples_k[j]:05d}.{extension}")
                    if not os.path.exists(sample_path):
                        continue
                    img, _ = self.__load_sample(sample_path)
                    if X_data is None:
                        X_data = np.expand_dims(img, axis=0)
                        Y_data = to_categorical([label], self.n_classes)
                    else:
                        X_data = np.vstack((X_data, np.expand_dims(img, axis=0)))
                        Y_data = np.vstack((Y_data, to_categorical([label], self.n_classes)))

            elif self.mode == 'testing':
                img, _ = self.__load_sample(ID)
                X_data[i, ] = np.expand_dims(img, axis=0)

        if self.mode == 'testing':
            return np.divide(X_data, 255)
        else:
            return np.divide(X_data, 255), Y_data


    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # '/root/germany/20/00001_00019.ppm'
    def __parse_datapath(self, ID):
        """Parses root image folder, country sample, label, sample number and extension from ID
        :param ID: absolute path to sample image
        :return: root image folder, country sample, label, sample number and extension
        """
        tmp = ID.split(os.sep)
        country = tmp[-3]
        label = int(tmp[-2])
        root_fldr = ID.split(os.sep + tmp[-2])[0]
        tmp = tmp[-1].split('.')
        extension = tmp[-1]
        tmp = tmp[0].split('_')
        sample = int(tmp[0])

        return root_fldr, country, label, sample, extension

    def __load_sample(self, img_file):
        img = Image.open(img_file)
        img = img.resize(self.dim)
        stat = ImageStat.Stat(img)
        brightness = stat.median[0]
        img = img.convert(mode='L')
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)
        return img, brightness

    def __augmentations(self, img, country, brightness_level):
        X = np.expand_dims(img, axis=0)
        for augmentation in self.aug:
            if country == 'germany' and (augmentation == zoom_in or augmentation == zoom_out):
                X = np.vstack((X, np.expand_dims(img, axis=0)))
                continue
            if (brightness_level < 60 and augmentation == brightness_dec) or \
                    (brightness_level > 180 and augmentation == brightness_inc):
                X = np.vstack((X, np.expand_dims(img, axis=0)))
                continue
            tmp = augmentation(img)
            tmp = np.expand_dims(tmp, axis=0)
            X = np.vstack((X, tmp))
        return X
