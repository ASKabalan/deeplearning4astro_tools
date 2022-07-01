import numpy as np
import random

class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """
    def __init__(self, X_array, y_array, size_of_augmented_data,augmentation_tuple):
        self.X_array            = X_array
        self.y_array            = y_array
        self.nb_examples        = size_of_augmented_data*len(X_array)

        augmentation_roulette_arr = []
        if (augmentation_tuple[0]):
            augmentation_roulette_arr.extend(['flip_x','flip_y'])
        if (augmentation_tuple[1]):
            augmentation_roulette_arr.extend(['rot_r','rot_l'])
        if (augmentation_tuple[2]):
            augmentation_roulette_arr.extend(['roll_r','roll_l'])

        self.augment_roulette   = np.array(augmentation_roulette_arr)

    def _augment_array(self,a,nb_elem):
        times_to_tile  =  nb_elem // len(a) 
        elem_to_concat =  nb_elem % len(a) 
        return np.concatenate([np.tile(a,times_to_tile) , a[0:elem_to_concat]])

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        It uses the augmentation parameters to return possiblt an augmented dataset
        

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """

        # how does this work
        # For example :
        #    If we request .1 validation in dataset [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9]

        #    first we split 
        #        possible_train_indices = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ]
        #        possible_valid_indices = [9]

        #    then if we ask for 8 examples we will have 

        #        train_indices =  [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7]
        #        valid_indices =  [9]

        #    If we ask for 15 examples we will have an augmented array repeating itself :

        #        train_indices =  [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 0 , 1 , 2 , 3 , 4 , 5 , 6]
        #        valid_indices =  [9]

        #    with augmentation we have a very high chance that elements do not repeated them self if the nb_examples is less that 3*orignal size
        #    the chance is very high if we have 5 times the original size
        #    the chance is 100% if we have 6 times !!
        #    TODO add more augmentations and ensure that unique elements and not repeated to avoid overfit 

        nb_of_train_indices = int((1 - valid_ratio) * len(self.X_array))

        data_set_indices = np.arange(len(self.X_array))
        possible_train_indices = data_set_indices[0:nb_of_train_indices]
        possible_valid_indices = data_set_indices[nb_of_train_indices:]

        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid

        train_indices = self._augment_array(possible_train_indices,nb_train)
        valid_indices = self._augment_array(possible_valid_indices,nb_valid)

        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:

            X = self.X_array[indices]
            y = self.y_array[indices]
            # converting to float needed?
            X = np.array(X, dtype='float32')
            y = np.array(y, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):

                X_batch = [np.expand_dims(img, -1)
                           for img in X[i:i + batch_size]]
                y_batch = [np.expand_dims(seg, -1)
                           for seg in y[i:i + batch_size]]

                for j in range(len(X_batch)):

                    # TODO more augments and ensure that there are only unique elements using a dict

                    # 1/6 chance for each augmentation and 1/6 that no augmentation at all
                    augmentation_roulette = None
                    if (len(self.augment_roulette) != 0):
                        augmentation_roulette = random.choice(self.augment_roulette)

                    if augmentation_roulette == 'flip_x':
                        X_batch[j] = np.flip(X_batch[j], axis=0)
                        y_batch[j] = np.flip(y_batch[j], axis=0)

                    elif augmentation_roulette == 'flip_y':
                        X_batch[j] = np.flip(X_batch[j], axis=1)
                        y_batch[j] = np.flip(y_batch[j], axis=1)

                    elif augmentation_roulette == 'rot_r':
                        X_batch[j] = np.rot90(X_batch[j], 1, axes=(0,1))
                        y_batch[j] = np.rot90(y_batch[j], 1, axes=(0,1))

                    elif augmentation_roulette == 'rot_l':
                        X_batch[j] = np.rot90(X_batch[j], -1, axes=(0,1))
                        y_batch[j] = np.rot90(y_batch[j], -1, axes=(0,1))

                    elif augmentation_roulette == 'roll_r':
                        X_batch[j] = np.roll(X_batch[j], 30)
                        y_batch[j] = np.roll(y_batch[j], 30)

                    elif augmentation_roulette == 'roll_l':
                        X_batch[j] = np.roll(X_batch[j], -30)
                        y_batch[j] = np.roll(y_batch[j], -30)


                yield np.array(X_batch), np.array(y_batch)
