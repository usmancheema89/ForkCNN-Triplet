import os
import numpy as np
import Data_loader.my_Augment as myaug
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def setsize(images):
    print(images.shape)
    resized_images = []
    for image in images:
        img = np.array(image)
        resized_images.append(resize(img, (img.shape[0] // 2, img.shape[1] // 2),anti_aliasing=True))
    resized_images = np.array(resized_images)
    print(resized_images.shape)
    return resize_img

def read_numpy(data_path, database, modalities):
    # Data
    # Data Path:  directory path to your Numpy Data Folder : r'E:\Work\Multi Modal Face Recognition\Numpy Data'
    # Database: string name of DB: 'IRIS' / 'I2BVSD'/ 'VISTH' 
    # Modalities: string names of modalities you want in a list ['Vis','The']  / ['Vis']
    # returns a dict with all images arrays with modality keyword
    image_data = dict()
    db_folder = database + ' Data'
    data_path = os.path.join(data_path,db_folder)

    for modality in modalities:
        mod_numpy_name = database + ' ' + modality + ' ' + 'Images.npy'
        mod_numpy_path = os.path.join(data_path, mod_numpy_name)
        image_data[modality] = np.load(mod_numpy_path)

    label_name = database + ' ' + 'Labels.npy'
    label_path = os.path.join(data_path, label_name)

    image_data['labels'] = to_categorical( np.load(label_path) )

    return image_data


def augment_data(db_data):
    aug_image_data = []      # contains data of one modality
    for mod_image_array in image_data: # for length of image_data [number of modalities]
        mod_aug_images = []
        mod_aug_labels = []
        img_count = 0
        for image in mod_image_array:  # for a single image in image array of a modality
            aug_images = myaug.augment_img(image)
            if len(mod_aug_images) == 0:
                mod_aug_images = aug_images
            else:
                mod_aug_images = [*mod_aug_images, *aug_images]
            for i in range(len(aug_images)):
                mod_aug_labels.append(labels[img_count])
            img_count += 1
        print(np.array(mod_aug_images).shape)
        aug_image_data.append(np.array(mod_aug_images))

    # print(len(aug_image_data))
    # print(aug_image_data[0].shape)
    # print(aug_image_data[1].shape)
    # print(np.array(mod_aug_labels).shape)
    return aug_image_data, np.array(mod_aug_labels)


def train_testSplit(db_data,db_name):
    # takes a dictionary with image data (numpy array) against modality key
    # and labels and "labels" key
    split_db = dict()
    test_db = dict()
    for key in db_data:
        if key is not 'labels':

            img_train, img_test, y_train, y_test = train_test_split(db_data[key], db_data['labels'], 
                test_size = 0.2,random_state = 42, stratify = db_data['labels'])
            keys = ['_img_train', '_img_test','_y_train','_y_test']
            split_db[key+keys[0]] = img_train
            split_db[key+keys[1]] = img_test
            split_db[keys[2]] = y_train
            split_db[keys[3]] = y_test
    # save_test_Data(test_db,db_name)
    # Sanity Check
        # if np.array_equal(split_db['The_y_train'],split_db['Vis_y_train']):
        #     print('y_train match')
        # if np.array_equal(split_db['The_y_test'],split_db['Vis_y_test']):
        #     print('y_test match')

    return split_db


def get_data(data_path, database, modalities):
    # datapath: path to numpy folder e.g E:\Work\Multi Modal Face Recognition\Numpy Data
    # database: one of 'IRIS', 'I2BVSD', 'VISTH'
    # Modalities: [Vis, The] one or two variable list

    image_data = read_numpy(data_path, database, modalities)
    split_db_data = train_testSplit(image_data,database)
    # image_data, labels = augment_data(image_data,labels)
    # Dictionary with data key: Vis_img_train, Vis_img_test
    # label key: _y_train, _y_test
    return split_db_data

def save_test_Data(test_db,db_name):
    with open('Pathfile.txt', 'r') as myfile:
        filepath = myfile.read().replace('\n','')
        for key in test_db:
            if '_test' in key:
                s_name = os.path.join(filepath, 'TestData', db_name + key +'.npy')
                if os.path.exists(s_name):
                    print(s_name, 'Img test data already exits')
                else:
                    np.save(s_name,test_db[key])


