from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt

def create_Generator(data_dic,batch_size):

    # create a data generator, fit it, flow (train,val), 
    mod_gen_train = dict()
    mod_gen_val = dict()
    dire = r'E:\Generated_Data'
    for key in data_dic:
        if '_img_train' in key:
            temp_gen = new_Generator()
            temp_gen.fit(data_dic[key])
            temp_train = temp_gen.flow( data_dic[key],data_dic['_y_train'],batch_size = batch_size, shuffle = False,subset = 'training',seed=42)
            # ,save_to_dir=dire, save_prefix=key, save_format='png' )
            temp_val = temp_gen.flow( data_dic[key],data_dic['_y_train'],batch_size = batch_size, shuffle = False,subset = 'validation',seed = 42)
            mod_gen_train[key] = temp_train
            mod_gen_val[key.replace('train','val')] = temp_val
            data_dic[key.replace('train','test')] = np.float32(data_dic[key.replace('train','test')])
            temp_gen.standardize(data_dic[key.replace('train','test')]) # = standardize_test(data_dic[key.replace('train','test')],temp_gen)
    return mod_gen_train, mod_gen_val

def standardize_test(data,temp_gen):
    for i in range(data.shape[0]):
        temp_gen.standardize(data[i])
    return data

def new_Generator():
    
    datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # zca_epsilon= 1e-6,
    # zca_whitening = True,
    fill_mode = 'constant',
    cval = 0,
    rotation_range=5, #degree of random rotations
    width_shift_range=0.1, # float: fraction of total width, if < 1,
    height_shift_range=0.1,# float: fraction of total heing, if < 1,
    brightness_range = [0.9,1.1], # Tuple or list of two floats. Range for picking a brightness shift value from
    shear_range = 0.0, # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    zoom_range = [0.8, 1.2], # Float or [lower, upper]. Range for random zoom
    vertical_flip=False,
    horizontal_flip=True,
    # rescale = 1.0/255,
    validation_split = 0.2, # Float Fraction of images reserved for validation
    # preprocessing_function = setsize
    # dataformat
    )

    return datagen



def multistream_Generator(data_dic,batch_size):
    
    mod_gen_train,mod_gen_val  = create_Generator(data_dic,batch_size)         # dictionary for all individual generators: modality, train, validate
    data_gen_train = JoinedGen(mod_gen_train)
    data_gen_val = JoinedGen(mod_gen_val)
    return data_gen_train, data_gen_val

def write_batch(images,labels,itr):
    for i in range(len(images)):
        img_name = r'E:\Generated_Data\image' + str(itr) + '_' + str(i) + '_' + str( np.argmax(labels[i]) ) + '.jpg'
        plt.imsave(img_name,images[i])
    input("Batch Done")

class JoinedGen(Sequence):
    def __init__(self, multi_modal_gen):
        self.gen = multi_modal_gen

    def __len__(self):
        return len(self.gen[list(self.gen)[0]])

    def __getitem__(self, i):
        x_batch = []
        if len(self.gen) == 1:
            x, y = self.gen[list(self.gen)[0]][i]
            # write_batch(x.astype(np.uint8),y,i)
            return x, y
        else:
            for key in self.gen:
                x, y = self.gen[key][i]
                x_batch.append(x)
            return x_batch, y

    # def next(self, i):
    #     x_batch = []
    #     if len(self.gen ==1):
    #         x, y = self.gen[key][i]
    #         return x, y
    #     else:
    #         for key in self.gen:
    #             x, y = self.gen[key][i]
    #             x_batch.append(x)
    #         return x_batch, y

    def on_epoch_end(self):
        for key in self.gen:
            self.gen[key].on_epoch_end()


