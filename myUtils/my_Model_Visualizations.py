################### VISUALIZING FILTERS AND MORE MAYBE ###################
from tensorflow.keras import models, Model
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot
import os



def get_Model_inter(m_path,output_name):
    base_model = models.load_model(m_path)
    my_input = base_model.inputs
    if len(output_name) == 2:
        my_output = [base_model.get_layer(output_name[0]).output, base_model.get_layer(output_name[1]).output]
    elif len(output_name) == 1:
        my_output = base_model.get_layer(output_name[0]).output
    model_inter = Model(my_input, my_output)
    return model_inter

def plot_features(featuresmap):

    if isinstance(featuresmap, list): 
        features = np.asarray(featuresmap)
        features = np.squeeze(features,axis = 1)  
    else:
        features = featuresmap
    
    sets, _, _, z = features.shape

    if z == 64:
        x,y = 8,8
    elif z == 128:
        x,y = 8,16
    elif z == 256:
        x,y = 16,16
    elif z == 512:
        x,y = 16,32
    
    for i in range(sets):
        idx = 1
        for _ in range(x):
            for _ in range(y):
                # specify subplot and turn of axis
                ax = pyplot.subplot(x, y, idx)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(features[i, :, :, idx-1], cmap='gray')
                idx += 1
                # show the figure
        pyplot.show()
        # pyplot.savefig('vis.png')


model_name = r'202004252238_vgg16_IRIS_2_Vis-The-_50_concatenate'
model_path = os.path.join(r'E:\Work\Multi Modal Face Recognition\Output\Models',model_name)
# multi_output_name = ['conv2_2_visible_stream','conv2_2_thermal_stream'] # VGG16 30
multi_output_name = ['conv3_2_visible_stream','conv3_2_thermal_stream'] # VGG16 50

# uni_output_name = ['conv3_1_merged'] # VGG16 30
uni_output_name = ['conv3_3_merged'] # VGG16 50

model_inter = get_Model_inter(model_path,uni_output_name)

v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Vis Images.npy')
t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS The Images.npy')

img = [np.expand_dims(v_data[200],axis=0), np.expand_dims(t_data[200],axis=0)]

featuremaps = model_inter.predict(img)
# (2,128,128,128)
plot_features(featuremaps)


