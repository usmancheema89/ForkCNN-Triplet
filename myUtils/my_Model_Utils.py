from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from math import exp
from datetime import datetime
import json,os

def get_metrics():
    acc = 'accuracy'
    auc = metrics.AUC(num_thresholds=200, curve='ROC', name='auc', thresholds=None, multi_label=False)
    fp = metrics.FalsePositives(thresholds=[0.001, 0.01, 0.1, 1.0], name='FP')
    tp = metrics.TruePositives(thresholds=[0.001, 0.01, 0.1, 1.0], name='TP')

    return [acc]#, auc, fp, tp]

def get_optimizer():
    optimizer = optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    # optimizer = optimizers.SGD(learning_rate=0.003)
    return optimizer

def get_callbacks(model_name):
    filepath = ''
    with open('Pathfile.txt', 'r') as myfile:
        filepath = myfile.read()
        filepath = filepath.split("\n")[0]
    tb_log_dir = os.path.join(filepath,'Logs', model_name)
    lg_log_dir = os.path.join(filepath,'History', model_name+'.csv')
    
    lg = callbacks.CSVLogger(lg_log_dir, separator=',', append=False)
    es = callbacks.EarlyStopping(monitor = 'loss', min_delta=0.0001, patience=40, verbose=1, mode='auto', restore_best_weights=True)
    # lr = callbacks.LearningRateScheduler(scheduler, verbose=1)
    #callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
    rop = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, verbose=1, mode='auto',min_delta=0.001, cooldown=0, min_lr=0.00000001)
    tb = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=False, write_images=False,update_freq='epoch', profile_batch=0) # embeddings_freq=0,embeddings_metadata=None)
    return [es, rop, tb, lg]

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0005
    else:
        return 0.0001 * exp(0.1 * (10 - epoch))

def get_name(db,mod,model,stream,mrg_p,mrg_s,editparam):
    #    'vgg16_IRIS_1_Vis-Ir_30_concatenate_H-M-S'
    u = '_'
    modalities = ''
    now = datetime.now()
    time = now.strftime(r"%Y%m%d%H%M")
    for modality in mod:
        modalities = modalities + modality + '-'
        
    name = time.replace(':','') +u+  model +u+ db +u+ str(stream) +u+ modalities +u+ str(mrg_p) +u+ mrg_s + editparam
    print("Training: ", name)
    return(name)

def save_model(model_name,model):
    filepath = ''
    with open('Pathfile.txt', 'r') as myfile:
        filepath = myfile.read()
        filepath = filepath.split("\n")[0]
    fullpath = os.path.join(filepath, 'Models', model_name)
    print('Saving Model to: ', fullpath)
    model.save(fullpath, overwrite = True)

