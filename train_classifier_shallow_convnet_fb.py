import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import Model, Sequential, load_model
from keras.layers import Dense,BatchNormalization,AveragePooling2D,MaxPooling2D,MaxPooling3D, \
    Convolution2D,Activation,Flatten,Dropout,Convolution1D,Reshape,Conv3D,TimeDistributed,LSTM,AveragePooling3D, \
    Input, AveragePooling3D, MaxPooling3D, concatenate, LeakyReLU, AveragePooling1D
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

def build_crops(X, y, increment, training=True):
    print("Obtaining sliding window samples (original data)")
    tmaximum = 500
    tminimum = 0
    X_list = []
    samplingfreq = 2
    
    while (tmaximum<=1116):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1116:
            break
    
    tmaximum = 501
    tminimum = 1
    while (tmaximum<=1116):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1116:
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
    #X_train = np.split(X_train, [1,2,3], axis=4)
    #X_val = np.split(X_val, [1,2,3], axis=4) 
    
    n_epoch = 500
    early_stopping = 20
    classes_len = len(np.unique(y_train))

    Y_train = to_categorical(y_train, classes_len)
    Y_val = to_categorical(y_val, classes_len)
    output_dim = classes_len
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    
    inputs = Input(shape=(X_shape[1],X_shape[2],X_shape[3],X_shape[4]))
    
    def layers(inputs):
        #pipe = Conv3D(40, (25,1,1), strides=(1,1,1), activation='linear')(inputs)
 
        pipe1 = Conv3D(40, (1,3,3), strides=(1,1,1), padding='same')(inputs)
        pipe1 = BatchNormalization()(pipe1)
        pipe1 = LeakyReLU(alpha=0.05)(pipe1)
        pipe1 = Dropout(0.5)(pipe1)
        pipe1 = AveragePooling3D(pool_size=(1,3,3), strides=(1,1,1), padding='same')(pipe1)
        #pipe1 = Conv3D(4, (1,1,1), strides=(1,1,1), padding='valid')(pipe1)
        #pipe1 = Reshape((pipe1.shape[1].value, 42, 4))(pipe1)
        
        pipe2 = Conv3D(40, (1,3,3), strides=(1,1,1), padding='same')(inputs)
        pipe2 = BatchNormalization()(pipe2)
        pipe2 = LeakyReLU(alpha=0.05)(pipe2)
        pipe2 = Dropout(0.5)(pipe2)
        pipe2 = Conv3D(40, (1,3,3), strides=(1,1,1), padding='same')(pipe2)
        pipe2 = BatchNormalization()(pipe2)
        pipe2 = LeakyReLU(alpha=0.05)(pipe2)
        pipe2 = Dropout(0.5)(pipe2)
        #pipe2 = Reshape((pipe2.shape[1].value, 42, 4))(pipe2)
        
        pipe12 = concatenate([pipe1,pipe2], axis=4)
        pipe12 = Conv3D(40, (1,6,7), strides=(1,1,1), padding='valid')(pipe12)
        pipe12 = BatchNormalization()(pipe12)
        pipe12 = LeakyReLU(alpha=0.05)(pipe12)
        pipe12 = Dropout(0.5)(pipe12)
        pipe12 = Conv3D(4, (1,1,1), strides=(1,1,1), padding='valid')(pipe12)
        pipe12 = Reshape((pipe12.shape[1].value, 4))(pipe12)
        
        pipe3 = Conv3D(40, (1,6,7), strides=(1,1,1), padding='valid')(inputs)
        pipe3 = BatchNormalization()(pipe3)
        pipe3 = LeakyReLU(alpha=0.05)(pipe3)
        pipe3 = Dropout(0.5)(pipe3)
        pipe3 = Conv3D(4, (1,1,1),  strides=(1,1,1), padding='valid')(pipe3)
        pipe3 = Reshape((pipe3.shape[1].value, 4))(pipe3)
        
        pipe = concatenate([pipe12,pipe3], axis=2)
        pipe = AveragePooling1D(pool_size=(75), strides=(15))(pipe)
        pipe = Flatten()(pipe)
        return pipe
    
    pipeline = layers(inputs)
    """
    pipeline = Dense(128)(pipeline)
    pipeline = BatchNormalization()(pipeline)
    pipeline = LeakyReLU(alpha=0.05)(pipeline)
    pipeline = Dropout(0.5)(pipeline)
    pipeline = Dense(64)(pipeline)
    pipeline = BatchNormalization()(pipeline)
    pipeline = LeakyReLU(alpha=0.05)(pipeline)
    pipeline = Dropout(0.5)(pipeline)
    """
    output = Dense(output_dim, activation=activation)(pipeline)
    model = Model(inputs=inputs, outputs=output)

    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    cb = [callbacks.ProgbarLogger(count_mode='samples'),
          callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=7,min_lr=0.00001),
          callbacks.ModelCheckpoint('./model_results_fb/A0{:d}_model.hdf5'.format(subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='val_acc', min_delta=0.0001)]
    model.summary()
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
              batch_size=128, epochs=n_epoch, verbose=1, callbacks=cb)



def evaluate_model(X_test, y_test, subject, crops):
    
    #X_test = np.split(X_test, [1,2,3], axis=4)
    
    all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
    actual = [ all_classes[i] for i in y_test ]
    #actual = np.hstack([ [i]*crops for i in actual ]) # Uncomment to enable crop-based testing
    
    num_trials = int(len(X_test)/crops)
    predicted = []
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    model = load_model('./model_results_fb/{}.hdf5'.format(model_name))
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
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    out_df.to_csv('./model_results_fb/{}.csv'.format(model_name))
    
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
        X, y, crops = build_crops(X, y, 12, training=True)
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
            X_test, y_test, crops = build_crops(X_test, y_test, 12, training=False)
            evaluate_model(X_test, y_test, i+1, crops)
            del(X_test)
            del(y_test)
            gc.collect()