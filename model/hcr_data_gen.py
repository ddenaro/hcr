# import the necessary libraries to access/pre-process your database and prepare input vectors for your Keras network
#######################################################################################################################
import keras
import numpy as np
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

###############################################################################################
# wrapper function
###############################################################################################
def create_test_generator(quant_test_set_dir, evaluation_test_set_dir, quant_test_ratio, batch_size):

    ########## MNIST example
    x_testset,y_testset = load_emnist()

    # quantization test-set
    quant_test_set_size = int(np.floor(len(x_testset)*quant_test_ratio))
    np.random.seed(10)
    random_partition_quant_test_set = np.random.permutation(len(x_testset))
    list_idx_input_quant = random_partition_quant_test_set[:quant_test_set_size]

    ####################################################################################################
    # Example with GenericInputBatchGenerator
    ####################################################################################################
    quant_datagen = GenericInputBatchGenerator(input=x_testset[list_idx_input_quant], labels=y_testset[list_idx_input_quant],
                                              batch_size=batch_size, shuffle=True)
    quant_test_set_labeled = quant_datagen.flow_labeled()
    # quant_test_set_labeled = None if no labels
    quant_test_set_unlabeled = quant_datagen.flow_unlabeled()
    quant_nsteps = len(quant_datagen)

    ####################################################################################################
    # Example with keras API ImageDataGenerator, very practical for 4D tensors inputs (batch, H, W, C)
    ####################################################################################################
    # quant_datagen = ImageDataGenerator(validation_split=quant_test_ratio-0.0001)
    # quant_test_set_labeled = quant_datagen.flow(x_testset, y_testset, batch_size=batch_size, subset="validation")
    # #quant_test_set_labeled = None if no labels
    # quant_test_set_unlabeled = quant_datagen.flow(x_testset, batch_size=batch_size, subset="validation")
    # quant_nsteps = len(quant_test_set_unlabeled)

    # evaluation test-set

    ####################################################################################################
    # Example with GenericInputBatchGenerator
    ####################################################################################################
    list_idx_input_eval = []
    [list_idx_input_eval.append(idx) for idx in np.arange(len(x_testset)) if idx not in list_idx_input_quant]
    eval_datagen = GenericInputBatchGenerator(input=x_testset[list_idx_input_eval], labels=y_testset[list_idx_input_eval],
                                               batch_size=batch_size, shuffle=True)
    eval_test_set_labeled = eval_datagen.flow_labeled()
    # eval_test_set_labeled = None if no labels
    eval_nsteps = len(eval_datagen)


    return quant_test_set_labeled, quant_test_set_unlabeled, quant_nsteps, eval_test_set_labeled, eval_nsteps


class GenericInputBatchGenerator:

    def __init__(self, input, labels, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.x = input
        self.y = labels
        self.shuffle = shuffle
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # Example from Keras site (https://keras.io/models/sequential/)in case data needs to be read from files
    # def generate_arrays_from_file(path):
    #     while True:

    # with open(path) as f:
    #     for line in f:
    #         # create numpy arrays of input data
    #         # and labels, from each line in the file
    #         x1, x2, y = process_line(line)
    #         yield ({'input_1': x1, 'input_2': x2}, {'output': y})

    def flow_labeled(self):
        while True:
            for idx in range(0, self.__len__()):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                # example: batch_x[i,] = np.load('data/' + idx * self.batch_size + '.npy') in case need to read in file
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                yield (np.array(batch_x), np.array(batch_y))

    def flow_unlabeled(self):
        while True:
            for idx in range(0, self.__len__()):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                # example: batch_x[i,] = np.load('data/' + idx * self.batch_size + '.npy') in case need to read in file

                yield (np.array(batch_x))

def load_emnist():
    
    # this dataset containts digits ( 0-9 ) and letters ( A-Z and a-z )
    # labels are (0-9) (10-35) (36-)
    emnist = scipy.io.loadmat("emnist-balanced.mat")
    x_test  = emnist["dataset"][0][0][1][0][0][0].astype(np.float32)
    y_test  = emnist["dataset"][0][0][1][0][0][1]

    # normalize data to 0.0 - 1.0 range
    x_test /= 255.0

    # reshape images and labes
    x_test_ds  = x_test.reshape(x_test.shape[0], 28, 28, 1, order="F")
    y_test_ds  = y_test.reshape(y_test.shape[0])

    # NUMBERS and CAPITAL LETTERS
    x_test  = x_test_ds[ np.nonzero(y_test_ds < 36) ]
    y_test  = y_test_ds[ np.nonzero(y_test_ds < 36) ]
    return x_test, to_categorical(y_test, 36)


