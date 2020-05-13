

# from tensorflow.keras import models, Model
# from tensorflow.keras.layers import Dense, Activation, Lambda
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# import tensorflow_addons as tfa
# from tensorflow.math import l2_normalize
# import io

# m_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\202004250436_vgg16_IRIS_2_Vis-The-_30_concatenate'
# o_model = models.load_model(m_path)
# output = Dense(4096, name='fc7')(o_model.layers[-5].output)
# # output = Activation('relu', name='fc7/relu')(output)
# output = Dense(1024, name='fc8')(output)
# output = Lambda(lambda x: l2_normalize(x,axis=1))(output)
# n_model = Model(o_model.inputs,output)
# n_model.compile(optimizer= 'SGD' , loss = tfa.losses.TripletSemiHardLoss())
# # print(n_model.get_layer(name = 'conv3_1_merged').get_weights())
# n_model.summary()
# v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Vis Images.npy')
# t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS The Images.npy')
# label =  np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Labels.npy')

# n_model.fit(x = [v_data,t_data],y = label, batch_size=32, epochs = 5, verbose = 1)

# results = n_model.predict([v_data,t_data])
# np.savetxt("vecs.tsv", results, delimiter='\t')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for labels in label:
#     out_m.write(str(labels) + "\n")
# out_m.close()
import csv, os
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.math import l2_normalize
import Data_loader.my_Get_DB as getDb
import Data_loader.my_Generator as myGen


def train_Model(model,db)
    data_path = r'E:\Work\Multi Modal Face Recognition\Numpy Data'
    data_dic = getDb.get_data(data_path, db, ['Vis', 'The'])
    ########### DATA AUGMENTATION ###########
    data_gen_train, data_gen_val  = myGen.multistream_Generator(data_dic,batch_size)

    

def add_Triplet(model):

    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if isinstance(layer, keras.layers.Flatten):
            output_layer = layer
            break


    output = keras.layers.Dense(4096, name='fc2')(layer.output)
    output = keras.layers.Dense(1024, name='fc1')(output)
    output = keras.layers.Lambda(lambda x: l2_normalize(x,axis=1),name = 'lambda')(output)
    
    model = keras.Model(model.inputs,output)
    model.compile(optimizer= 'SGD' , loss = tfa.losses.TripletSemiHardLoss())
    # print(n_model.get_layer(name = 'conv3_1_merged').get_weights())
    return model

def get_Model(archi, database, mergeat, mergewith, train):
    with open('Pathfile.txt', 'r') as myfile:
        output_path = myfile.read().replace('\n','')
    model_dir = os.path.join(output_path,'Models')
    namecheck = [archi, database, mergeat, mergewith, train]
    for model_name in os.listdir(model_dir):
        if all(x in model_name for x in namecheck):
            model = keras.models.load_model( os.path.join(model_dir, model_name) )
            return model, model_name

def read_train_CSV():
    with open('TripletLossTrain.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            archi, database, mergeat, mergewith, train = row
            if train == '1':
                print(archi, database, mergeat, mergewith, train)
                model, model_name = get_Model(archi, database, mergeat, mergewith, train)
                model = add_Triplet(model)
                model = train_Model(model, database)

read_train_CSV()