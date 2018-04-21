import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import Model, Sequential, load_model
from keras.layers import Dense,BatchNormalization,AveragePooling2D,MaxPooling2D,MaxPooling3D, \
    Convolution2D,Activation,Flatten,Dropout,Convolution1D,Reshape,Conv3D,TimeDistributed,LSTM,AveragePooling3D, \
    Input, AveragePooling3D, MaxPooling3D, concatenate, LeakyReLU, AveragePooling1D, GlobalAveragePooling1D, \
    multiply
from keras.utils.np_utils import to_categorical
from keras import optimizers, callbacks
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import seaborn as sn
import read_bci_data_fb

'''
Training model for classification of EEG samples into motor imagery classes
'''
def se_block(input_tensor, compress_rate = 9):
    num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
    bottle_neck = int(num_channels//compress_rate)
 
    se_branch = GlobalAveragePooling1D()(input_tensor)
    se_branch = Dense(bottle_neck, activation='relu')(se_branch)
    se_branch = Dense(num_channels, activation='sigmoid')(se_branch)
 
    x = input_tensor 
    out = multiply([x, se_branch])
 
    return out

def build_crops(X, y, increment, training=True):
    print("Obtaining sliding window samples (original data)")
    tmaximum = 500
    tminimum = 0
    X_list = []
    samplingfreq = 2
    
    while (tmaximum<=1000):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1000:
            break
    
    tmaximum = 501
    tminimum = 1
    while (tmaximum<=1000):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1000:
            break
    
    crops = len(X_list)
    X = np.array(X_list)
    X = X.transpose(1,0,2,3,4,5)
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4],X.shape[5])    
    
    if training:
        y = [ y for l in range(crops)]
        y = np.stack(y, axis=-1)
        y = y.flatten()
    
    return X, y, crops

def train(X_train, y_train, X_val, y_val, subject):
    
    X_shape = X_train.shape
    X_val_shape = X_val.shape
    X_train = np.split(X_train, np.arange(1,9,1).tolist(), axis=4)
    X_val = np.split(X_val, np.arange(1,9,1).tolist(), axis=4) 
    
    n_epoch = 500
    early_stopping = 15
    classes_len = len(np.unique(y_train))

    Y_train = to_categorical(y_train, classes_len)
    Y_val = to_categorical(y_val, classes_len)
    output_dim = classes_len
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    
    train_inputs = []
    val_inputs = []
    inputs = []
    for i in range(len(X_train)):
        inputs.append(Input(shape=(X_shape[1],X_shape[2],X_shape[3],1)))
        train_inputs.append(X_train[i].reshape(X_shape[0],X_shape[1],X_shape[2],X_shape[3],1))
        val_inputs.append(X_val[i].reshape(X_val_shape[0],X_val_shape[1],X_val_shape[2],X_val_shape[3],1))
    
    def layers(inputs):
        pipe = Conv3D(40, (1,6,7), strides=(1,1,1), padding='valid')(inputs)
        pipe = LeakyReLU(alpha=0.05)(pipe)
        pipe = Dropout(0.5)(pipe)
        pipe = BatchNormalization()(pipe)
        #pipe = Reshape((pipe.shape[1].value, 40))(pipe)
        pipe = Reshape((pipe.shape[1].value, 1, 40, 1))(pipe)
        return pipe

    pipes = []
    for i in range(len(X_train)):
        pipes.append(layers(inputs[i]))
        
    pipeline = concatenate(pipes, axis=2)
    pipeline = Conv3D(4, (1,9,40), strides=(1,1,1), padding='valid')(pipeline)
    #pipeline = se_block(pipeline, compress_rate = 15)
    pipeline = LeakyReLU(alpha=0.05)(pipeline)
    pipeline = Dropout(0.5)(pipeline)
    pipeline = BatchNormalization()(pipeline)
    pipeline = AveragePooling3D(pool_size=(75,1,1), strides=(15,1,1))(pipeline)
    pipeline = Flatten()(pipeline)
    output = Dense(output_dim, activation=activation)(pipeline)
    model = Model(inputs=inputs, outputs=output)

    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    cb = [callbacks.ProgbarLogger(count_mode='samples'),
          callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=7,min_lr=0.00001),
          callbacks.ModelCheckpoint('./model_results_fb_global/A0{:d}_model.hdf5'.format(subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='val_acc', min_delta=0.0001)]
    model.summary()
    model.fit(train_inputs, Y_train, validation_data=(val_inputs, Y_val), 
              batch_size=128, epochs=n_epoch, verbose=1, callbacks=cb)



def evaluate_model(X_test, y_test, subject, crops):
    
    X_test = np.split(X_test, np.arange(1,9,1).tolist(), axis=4)
    
    all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
    actual = [ all_classes[i] for i in y_test ]
    #actual = np.hstack([ [i]*crops for i in actual ]) # Uncomment to enable crop-based testing
    
    num_trials = int(len(X_test[0])/crops)
    predicted = []
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    model = load_model('./model_results_fb_global/{}.hdf5'.format(model_name))
    y_pred = model.predict(X_test)
    #Y_preds = np.argmax(y_pred, axis=1)
    Y_preds = np.argmax(y_pred, axis=1).reshape(num_trials, crops)
    for j in Y_preds:
        (values,counts) = np.unique(j, return_counts=True)
        ind=np.argmax(counts)
        predicted.append(all_classes[values[ind]])

    kappa_score = metrics.cohen_kappa_score(actual, predicted, labels=all_classes)
    
    confusion_metric =  metrics.confusion_matrix(actual,predicted,labels=all_classes)
    clf_rep = metrics.precision_recall_fscore_support(actual, predicted)
    out_dict = {
         "precision" :clf_rep[0].round(3)
        ,"recall" : clf_rep[1].round(3)
        ,"f1-score" : clf_rep[2].round(3)
        ,"support" : clf_rep[3]
    }
    out_df = pd.DataFrame(out_dict, index = np.sort(all_classes))
    out_df['kappa'] = kappa_score
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 3) if x.name!="support" else  round(x.sum(), 3)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    out_df.to_csv('./model_results_fb_global/{}.csv'.format(model_name))
    
    print(metrics.classification_report(actual,predicted))
    print('kappa value: {}'.format(kappa_score))
    """
    fig = plt.figure(figsize = (10,7), dpi=100)
    ax = plt.subplot()
    sn.heatmap(confusion_metric, annot=True, ax = ax)
    ax.set_xlabel('Predicted Classes')
    ax.set_ylabel('True Classes')
    ax.set_title('Subject A0{:d} Confusion Matrix'.format(subject))
    ax.xaxis.set_ticklabels(all_classes)
    ax.yaxis.set_ticklabels(all_classes)
    #plt.show()
    fig.savefig('./model_results_fb/{}_cm.png'.format(model_name))
    #plt.clf()
    """

if __name__ == '__main__': # if this file is been run directly by Python
    
    # load bci competition data set
    
    raw_edf_train, subjects_train = read_bci_data_fb.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    raw_edf_test, subjects_test = read_bci_data_fb.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    # Iterate training and test on each subject separately
    for i in range(9):
        train_index = subj_train_order[i] 
        test_index = subj_test_order[i]
        np.random.seed(123)
        X, y = read_bci_data_fb.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
        X, y, crops = build_crops(X, y, 20, training=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        tf.reset_default_graph()
        with tf.Session() as sess:
            train(X_train, y_train, X_val, y_val, i+1)
            del(X_train)
            del(y_train)
            del(X_val)
            del(y_val)
            del(X)
            del(y)
            gc.collect()
            X_test, y_test = read_bci_data_fb.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
            X_test, y_test, crops = build_crops(X_test, y_test, 20, training=False)
            evaluate_model(X_test, y_test, i+1, crops)
            del(X_test)
            del(y_test)
            gc.collect()