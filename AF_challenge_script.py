#!/usr/bin/env python
# coding: utf-8

# To run this notebook, you will need to ensure your environment runs with at least Python 3.6 and has all the dependencies installed (see below). You will need to download a copy of the file "ResNet_30s_34lay_16conv.hdf5" from this GitHub respository: https://github.com/fernandoandreotti/cinc-challenge2017/tree/master/deeplearn-approach and store it in the same folder as this notebook. You will also need to download the file "training2017.zip" from this page: https://physionet.org/challenge/2017/ and unzip it into the same folder as this notebook (so that the .mat and .hea files are contained in a subdirectory named "training2017". Finally, you should create an empty subdirectory named "training_nps".
# 
# Once all this is done, you should be able to hit "Run All". The experiment was originally run on a machine with an Intel i7 processor, a NVIDIA GeForce GTX 1060 GPU and 8GB RAM. With these specifications, it takes about 12 hours to execute in full. Approximately 7GB of HDD space will be required for the ECG images. 
# 
# The output of the code will be approximately 110,000 images of ECG traces and an automated analysis system that can classify them into one of four cardiac rhythm groups with an F1 score in excess of 0.8 (as evaluated by 5-fold cross-validation, where no version of the model has been exposed to any data from patients whose ECGs it is tasked with analysing during the validation process).

# In[ ]:


import keras
from keras.models import load_model
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import glob, os
import cv2
from matplotlib.pyplot import figure
import pandas as pd
from IPython.display import clear_output
import numpy as np
from multiprocessing import Pool 
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from PIL import Image
from random import randint
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Input, Flatten, Dense, Conv2D, Activation, MaxPool2D,MaxPooling1D,Conv1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, Sequence
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import pickle


# The function below takes as input a directory containing .mat files with the raw data from single lead ECG recordings.
# The signal data takes the forms of a list of readings in mV at a frequency specificied by the FS (frames/second) value.
# 
# It transposes each of those records into a series of .png images where height = image_y and width = (image_y * ecg_seconds).
# 
# It returns a list of filenames for the created images in the following format: 'temp_ecgs/[original_record_name]\_[segment_number].png', where the original_record_name has had the '.mat' extesion removed. It also returns a list of the original record names along with a list of corresponding target labels.
# 
# The function can take a while to run, so its outputs of the function are saved to disk for future runs.

# In[ ]:


def create_ecg_images(image_y,ecg_seconds,FS, directory,sliding_window_seconds,max_seconds_of_padding,script_runs,records_per_batch):
    
    
    #Ensure the target directory for the images exists:
    dirName='temp_ecgs'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " created. Creating ECGs...") 
    except FileExistsError:
        print("Directory " , dirName ,  " exists. Continuing with next batch of ECG creation...")
    
    
    ecg_list=[] #Will be returned as a list of image file names for the ECGs of ecg_seconds long
    image_x = image_y * ecg_seconds #Calculate the image width based on its height. Note actual image size on save may vary substantially, will be rescaled on import.
    length=FS*ecg_seconds #Calculate how many values to include in each ECG segment (i.e. each image)
    image_list=[] #List of images already created (in case function was interrupted before all records processed)
    ground_truth=[] #List of labels
    original_ecg_list=[] #List of original record names (without '.mat' extension). Only records >= ecg_seconds included.
    ecgs_in_physionet=0 #Total number of records in source, for comparison with length of original_ecg_list (to see how many records were too short to be included).
    sliding_window_frames=sliding_window_seconds*FS
    max_frames_of_padding=max_seconds_of_padding*FS
    df = pd.read_csv(directory + "/REFERENCE.csv") #Load up label database.
    z=0 #Helper variable in case images need to be written in batches (can clog up memory of too many at once).
    for file in glob.glob("temp_ecgs/*.png"): #Loop to obtain list of images already created.
        image_name=file.split('.')[0]
        image_name=image_name.split('\\')[-1]
        image_list.append(image_name)
    for file in glob.glob(directory + "/*.mat"): #Main loop.
        if (z<records_per_batch+(script_runs*records_per_batch)): #Helper function for writing images in batches. Always initialise z=0 - files already written will be ignored.
            ecgs_in_physionet+=1 #Count how many records in source data.
            matlab_array_contents= sio.loadmat(file)
            ecg = matlab_array_contents['val'] #Load ECG values from '.mat' file.
            padding=np.zeros((1,max_frames_of_padding))
            ecg=np.concatenate((ecg,padding),axis=1)
            maxi=np.argmax(ecg[0,:])
            mini=np.argmin(ecg[0,:]) #Get minimum and maximum mV values for rescaling later.
            del matlab_array_contents #Save on memory.
            if(ecg.shape[1] >= length): #Ensure records is at least ecg_seconds long.
                segments=int((ecg.shape[1] - length) // sliding_window_frames) #Calculate how many segments can be gleaned from each record.
                #print('Length of ECG:',ecg.shape[1]/FS,'Segments:',segments)
                for k in range (segments):
                    savename = file.split('.')
                    savename = savename[0].split('\\')[-1] #Isolate record name without directory prefix and '.mat' extension.
                    ground_truth.append(df["label"][df["ECG"] == savename].values[0]) #Add corresponding label to ground_truth list.
                    if (savename not in original_ecg_list):
                        original_ecg_list.append(savename) #Add the record name to original_ecg_list
                    savename=savename+'_'+str(k).zfill(3) #Now adjust the filename to '[record_name]_[segment_number]'.
                    if (savename not in image_list): #Check image for this segment hasn't already been created.
                        savename = './temp_ecgs/' + savename + '.png'
                        ecg_list.append(savename) #Add image filename to ecg_list.
                        ecg_k=ecg - np.mean(ecg)
                        ecg_k=ecg_k/np.std(ecg_k)
                        ecg_k=np.interp(ecg_k, (ecg_k.min(), ecg_k.max()), (0, +1)) #These three lines for feature scaling.
                        ecg_k=ecg_k[0,k*sliding_window_frames:length+(k*sliding_window_frames)] #Select out segment from total list of ECGs values for this record.
                        fig, ax = plt.subplots()
                        ax.set(xlim=[0, length],ylim=[0,1])
                        ax.plot(ecg_k,'k')
                        DPI = fig.get_dpi()
                        fig.set_size_inches(image_x/DPI,image_y/DPI)
                        ax.set_axis_off()
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        fig.savefig(savename,facecolor='w',bbox_inches=extent)
                        plt.close(fig) #These nine lines create, save and close the images using standard tools from MatPlotLib.
                        del ax
                        del fig #Save memory.
                        del ecg_k
                    else:
                        savename = './temp_ecgs/' + savename + '.png'
                        ecg_list.append(savename) #If the image has already been created, simply add its filename to list.
            else:
                #print('This ECG is less than 30s!')
                k=0
                savename = file.split('.')
                savename = savename[0].split('\\')[-1] #Isolate record name without directory prefix and '.mat' extension.
                ground_truth.append(df["label"][df["ECG"] == savename].values[0]) #Add corresponding label to ground_truth list.
                if (savename not in original_ecg_list):
                    original_ecg_list.append(savename) #Add the record name to original_ecg_list
                savename=savename+'_'+str(k).zfill(4) #Now adjust the filename to '[record_name]_[segment_number]'.
                if (savename not in image_list): #Check image for this segment hasn't already been created.
                    savename = './temp_ecgs/' + savename + '.png'
                    ecg_list.append(savename) #Add image filename to ecg_list.
                    ecg_k=np.zeros(length)
                    ecg_temp=ecg - np.mean(ecg)
                    ecg_temp=ecg_temp/np.std(ecg_temp)
                    ecg_temp=np.interp(ecg_temp, (ecg_temp.min(), ecg_temp.max()), (0, +1)) #These three lines for feature scaling.
                    ecg_k[:ecg_temp.shape[1]]=ecg_temp[0,:] #Select out segment from total list of ECGs values for this record.
                    del ecg_temp
                    fig, ax = plt.subplots()
                    ax.set(xlim=[0, length],ylim=[0,1])
                    ax.plot(ecg_k,'k')
                    DPI = fig.get_dpi()
                    fig.set_size_inches(image_x/DPI,image_y/DPI)
                    ax.set_axis_off()
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(savename,facecolor='w',bbox_inches=extent)
                    plt.close(fig) #These nine lines create, save and close the images using standard tools from MatPlotLib.
                    del ax
                    del fig #Save memory.
                    del ecg_k
                else:
                    savename = './temp_ecgs/' + savename + '.png'
                    ecg_list.append(savename) #If the image has already been created, simply add its filename to list.
            del ecg
            z+=1
    print('Lists created! ECG_list length: ',len(ecg_list),' Truth list length: ',len(ground_truth))
    print('ECGs in list: ', len(original_ecg_list),' ECGs provided by Physionet: ',ecgs_in_physionet)
    with open('training_nps/ecg_list.p', 'wb') as list_file:
            pickle.dump(ecg_list, list_file)
    with open('training_nps/ground_truth.p', 'wb') as list_file:
            pickle.dump(ground_truth, list_file)
    with open('training_nps/original_ecg_list.p', 'wb') as list_file:
            pickle.dump(original_ecg_list, list_file)
    return [ecg_list,ground_truth,original_ecg_list]


# The function below returns the same values as the previous function, if the values have already been saved to disk.

# In[ ]:


def load_lists():
    print("Loading pickled lists...")
    with open('training_nps/ecg_list.p', 'rb') as list_file:
            ecg_list = pickle.load(list_file)
    with open('training_nps/ground_truth.p', 'rb') as list_file:
            ground_truth = pickle.load(list_file)
    with open('training_nps/original_ecg_list.p', 'rb') as list_file:
            original_ecg_list = pickle.load(list_file)
    print("Done!")
    return [ecg_list,ground_truth,original_ecg_list]


# The function below measures the mean, max and min mV values from the entire dataset. It can be useful if you want to rescale your data.

# In[ ]:


def get_ecg_ranges(directory):
    maxi=0
    mini=0
    maxis=0
    minis=0
    for file in glob.glob(directory + "/*.mat"):
        matlab_array_contents= sio.loadmat(file)
        ecg = matlab_array_contents['val'][0,:]
        ecg = ecg - np.mean(ecg)
        ecg = ecg/np.std(ecg)
        maxi+=np.amax(ecg)
        mini+=np.amin(ecg)
        maxis+=1
        minis+=1
    ave_max=maxi/maxis
    ave_min=mini/minis
    print('Average y_max: ', ave_max, ' Average y_min: ', ave_min)


# The function below takes an image array and loops through the columns looking for the row index of the first encountered black pixel. Once all row indices are logged, any missing values are filled in by averaging values from the row indicies in neighbouring columns. This data will later be treated as a surrogate for raw ECG data (i.e. in mV / [second / Hz]) in order to make use of previous work taking raw ECG data as input.
# 
# In our paper, the ECGs were  transposed to image files at a resolution that only preserved half the data (approximating the information loss one would expect if using an old desktop scanner). For this reason, the function below subsequently expands the generated image_x-dimensional vector by a factor of two using more nearest-neighbour averaging. (We wanted to use a pre-trained model including the final, fully-connected layer. Thus, we needed to preserve the length of the input feature vector.)
# 
# All values are rescaled between 0-1 using linear interpolation. The resulting 2(image_x)-dimensional vector forms the input for the ResNet convolutional neural network, whose architecture is based on recent research from Stanford University and the University of Oxford on rhythm recognition using deep learning.
# 
# Currently this function relies on a very clean dataset, because any artefactual black pixels or rotation of the source image would skew the results. There are several ways this process could be made more robust. However, our intention in conducting this study was test the null hypothesis that deep learning based ECG analysis using image inputs would result in catastrophic loss of classifier performance. Refining the system will no doubt form the basis of future work.

# In[ ]:


def manually_read_image(data):
    try:
        file=data[0]
        img=data[1]
        y_list=[]
        for column in range(img.shape[1]):
            switch = False
            for row in range(img.shape[0]):
                if (img[row,column]<0.2 and switch==False):
                    if (row==0):
                        y_list.append(1)
                        switch=True
                    else:
                        y_list.append(1-(row/img.shape[0]))
                        switch=True
                if (row==img.shape[0]-1 and switch==False):
                    y_list.append(None)
        y_list_expanded = []
        for y_index in range(len(y_list)):
            if (y_list[y_index]==None):
                index_adder=1
                if (y_index+index_adder >= len(y_list)):
                    y_list[y_index]=y_list[y_index-1]
                else:
                    next_y=y_list[y_index+index_adder]
                    while next_y==None:
                        index_adder+=1
                        if (y_index+index_adder >= len(y_list)):
                            if (y_index==0):
                                raise ValueError('There are no values in this ECG!')
                            else:
                                next_y=y_list[y_index-1]
                        else:
                           next_y=y_list[y_index+index_adder]
                    if(y_index==0 or y_index+index_adder >= len(y_list)):
                        y_list[y_index]=next_y
                    else:
                        y_list[y_index]=y_list[y_index-1]+((next_y-y_list[y_index-1])/(index_adder+1))
        for y_index in range(len(y_list)):
            y_list_expanded.append(y_list[y_index])
            if (y_index==len(y_list)-1):
                y_list_expanded.append(y_list[y_index])
            else:
                generated_y=y_list[y_index]+((y_list[y_index+1]-y_list[y_index])/2)
                y_list_expanded.append(generated_y)
        y_list_expanded=np.interp(y_list_expanded, (0, +1), (-5, +6)) #Rescales data to better reflect the data the Oxford model trained on
        return [y_list_expanded,file]
    except ValueError as e:
        print('Blank image in batch')


# The function below loads an image, rescales it to the target size and compresses it to greyscale. It returns a numpy array array for the greyscale image.
# 
# The name of the ECG file is passed as both input and output arguments. This is a simple way of keeping track of the images during multiprocessing.

# In[ ]:


def load_images(ecg_list):
    file=ecg_list[0]
    image_y=ecg_list[1]
    ecg_seconds=ecg_list[2]
    x_width=image_y*ecg_seconds
    try:
        x=Image.open(file).convert('L')
        # To save passing variables to multithreading processes, I've hard-coded image sizes into function:
        x=x.resize((x_width,image_y), Image.ANTIALIAS)
        x=np.asarray(x,dtype=np.float32)/255
        return [x,file]
    except SyntaxError as e:
        print('Failed to load image: ', file)


# The function below is part of the workflow of preprocessing image files into input vectors for the machine learning model. It exists as an independent function simply to capitalise on multiprocessing for computational efficiency.

# In[ ]:


def create_training_set(ecg_list):
    pool = ThreadPool(4) 
    results = pool.map(load_images, ecg_list)
    image_list=[]
    file_list=[]
    for i in range(len(results)):
        image_list.append(results[i][0])
        file_list.append(results[i][1])
    pool.close() 
    pool.join()
    del results
    return [image_list,file_list]


# The function below takes image files, preprocesses them and returns 'extrapolated' signal data as feature vectors for the machine learning model. The name of the function reflects the fact that the 'bottom layer' of this process is formed of rule-based computing techniques, where the 'top layer' is machine learning-based. The preprocessing that connects the two is, hence, the 'middle layer'.

# In[ ]:


def create_middle_layer(ecg_list,image_y,ecg_seconds,FS):
    creator_list=[]
    for ecg in ecg_list:
        temp_list=[ecg,image_y,ecg_seconds]
        creator_list.append(temp_list)
        del temp_list
    images,files=create_training_set(creator_list)
    del creator_list
    data=[]
    for i in range(len(images)):
        mini_data=[files[i],images[i]]
        data.append(mini_data)
    pool = ThreadPool(4) 
    results = pool.map(manually_read_image, data)
    del data
    pool.close() 
    pool.join()
    X=np.zeros((len(results),FS*ecg_seconds))
    ecgs=[]
    for i in range(len(results)):
        X[i,:]=results[i][0]
        ecgs.append(results[i][1])
    del results
    return X,ecgs


# The function below simply saves the data returned from the create_middle_layer function to avoid the need to reprocess all the images each time the script is run.

# In[ ]:


def save_training_array(ecg_list,image_y,ecg_seconds,FS,ground_truth,batch_number):
    print('Creating training data...')
    X,ecgs=create_middle_layer(ecg_list,image_y,ecg_seconds,FS)
    Y=np.zeros((X.shape[0],4))
    classes = ['A', 'N', 'O','~']
    new_ecg_list=[]
    for i in range(X.shape[0]):
        index=ecg_list.index(ecgs[i])
        y=label_to_one_hot(ground_truth[index])
        Y[i,:]=y
        new_ecg_list.append(ecgs[i])
    length=X.shape[0]
    print('Saving ',length,' training examples...')
    np.save('training_nps/x_train_'+str(batch_number)+'.npy', X)
    np.save('training_nps/y_train_'+str(batch_number)+'.npy', Y)
    with open('training_nps/original_record_'+str(batch_number)+'.p', 'wb') as list_file:
        pickle.dump(new_ecg_list, list_file)
    print('Done!')
    del X
    del Y


# The function below divides training data into batches. This experiment was original run on a machine with 8GB RAM, so a few workarounds like this were required to operate within hardware constraints.

# In[ ]:


def create_x_batches(ecg_list,image_y,ecg_seconds,FS,ground_truth,x,original_ecg_list):
    start=0
    for batch_number in range(x):
        if(os.path.isfile('training_nps/x_train_'+str(batch_number)+'.npy') is False or
           os.path.isfile('training_nps/Y_train_'+str(batch_number)+'.npy') is False or
           os.path.isfile('training_nps/original_record_'+str(batch_number)+'.p') is False):
            print('Generating batches from number',start,'of',x)
            break
        else:
            start=batch_number+1
    if (start == x):
        return
    else:
        slice_size=len(original_ecg_list)//x
        original_limiter_list=[] #list of the original Physionet record sitting at the start of each slice
        image_limiter_list=[] #(k+1)list of indices of the image at the start of each slice, with len(ecg_list) appended
        #Create original_limiter_list:
        for i in range(x):
            original_limiter_list.append(original_ecg_list[i*slice_size])
        #Create image_limiter_list:
        split_str=ecg_list[0].split('.')[1]
        split_str=split_str.split('/')[-1]
        split_str=split_str.split('_')[0]
        ecg_name=split_str
        last_index=0
        for ecg_index in range(len(ecg_list)):
            split_str=ecg_list[ecg_index].split('.')[1]
            split_str=split_str.split('/')[-1]
            split_str=split_str.split('_')[0]
            if (split_str != ecg_name): #Triggers every time an image belonging to a new ECG record is reached.
                if (split_str in original_limiter_list):
                    image_limiter_list.append(last_index) #If the new record is a slice limiter, add index to list.
                    last_index=ecg_index
                ecg_name=split_str #Otherwise, reset ecg_name so all images belonging to this record are ignored.
            if (ecg_index==len(ecg_list)-1):
                image_limiter_list.append(last_index)
                image_limiter_list.append(len(ecg_list)) #Append the length of ecg_list to the list.
        print('Image_limiter_list length: ',len(image_limiter_list),' Contents: ',image_limiter_list) #For debugging
        #Create and save data in batches:
        for i in range(start,x):
            start=image_limiter_list[i]
            end=image_limiter_list[i+1]
            print('Creating and saving batch ',i+1,'of ',x,'...')
            save_training_array(ecg_list[start:end],image_y,ecg_seconds,FS,ground_truth[start:end],i)
            print('Done with batch ',i+1,'!')


# Because the dataset is heavily skewed, F1 scoring is much more important than accuracy. The function below calculates the F1 score.

# In[ ]:


def F1_score(prediction,truth,F1_list):
    F1_list[truth]+=1
    F1_list[prediction+4]+=1
    if (prediction==truth):
        F1_list[prediction+8]+=1
    return F1_list


# The cell below contains two helper functions to translate one-hot vectors (returned by the machine learning model) back into human-readable labels and vice versa.

# In[ ]:


def one_hot_to_label(one_hot):
    y=np.argmax(one_hot)
    label=''
    if(y==0):
        label='AF'
    elif(y==1):
        label='NSR'
    elif(y==2):
        label='Other'
    elif(y==3):
        label='Noisy'
    return label

def label_to_one_hot(label):
    one_hot=np.zeros((1,4))
    if(label=='A'):
        one_hot[0,0]=1
    elif(label=='N'):
        one_hot[0,1]=1
    elif(label=='O'):
        one_hot[0,2]=1
    elif(label=='~'):
        one_hot[0,3]=1
    return one_hot


# In[ ]:


#Helper function for calculating F1 score
def quick_f1_function(start,F1_list):
    if (F1_list[start]+F1_list[start+4]==0 or F1_list[start+8]==0):
        f1=0
    else:
        f1=(2*F1_list[start+8])/(F1_list[start]+F1_list[start+4])
    return f1


# The function below makes predictions using a trained model and returns the F1 score for the batch.

# In[ ]:


def make_predictions(model,X,Y,new_ecg_list,batch_size):
    F1_list=[0,0,0,0,0,0,0,0,0,0,0,0] #A N O P a n o p Aa Nn Oo Pp
    y_pred_per_image = model.predict(X,batch_size=batch_size)
    total_predictions=y_pred_per_image.shape[0]
    del X
    #Check accuracy
    maxes=np.argmax(y_pred_per_image,axis=1)
    maxes_Y=np.argmax(Y,axis=1)
    comparison=maxes==maxes_Y
    accuracy=comparison.sum()/comparison.size
    del maxes_Y
    del comparison
    #Convert predictions to one-hot vectors:
    zeros=np.zeros((y_pred_per_image.shape[0],y_pred_per_image.shape[1]))
    zeros[np.arange(y_pred_per_image.shape[0]), maxes] = 1
    y_pred_per_image=zeros
    del maxes
    del zeros
    #Create a list of of total_predictions x 2, containing start and stop indices for each unique record:
    ecgs_from_same_record=[]
    split_str=new_ecg_list[0].split('.')[1]
    split_str=split_str.split('/')[-1]
    split_str=split_str.split('_')[0]
    ecg_name=split_str
    last_index=0
    for ecg_index in range(len(new_ecg_list)):
        split_str=new_ecg_list[ecg_index].split('.')[1]
        split_str=split_str.split('/')[-1]
        split_str=split_str.split('_')[0]
        if (split_str!=ecg_name):
            start_stop_index=[]
            start_stop_index.append(last_index)
            start_stop_index.append(ecg_index)
            last_index=ecg_index
            ecg_name=split_str
            ecgs_from_same_record.append(start_stop_index)
        if (ecg_index==len(new_ecg_list)-1):
            start_stop_index=[]
            start_stop_index.append(last_index)
            start_stop_index.append(len(new_ecg_list))
            ecgs_from_same_record.append(start_stop_index)
    y_pred=np.zeros((len(ecgs_from_same_record),4))
    y_true=np.zeros((len(ecgs_from_same_record),4))
    for i in range(len(ecgs_from_same_record)):
        a=ecgs_from_same_record[i][0]
        b=ecgs_from_same_record[i][1]
        y_pred[i,:]=np.sum(y_pred_per_image[a:b,:], axis=0)
        y_true[i,:]=Y[a,:]
    for i in range(y_pred.shape[0]):
        this_y=np.argmax(y_true[i,:])
        this_pred=np.argmax(y_pred[i,:])
        F1_list=F1_score(this_pred,this_y,F1_list)
    f1_a=quick_f1_function(0,F1_list)
    f1_n=quick_f1_function(1,F1_list)
    f1_o=quick_f1_function(2,F1_list)
    f1_total=(f1_a+f1_n+f1_o) / 3
    print('Accuracy: ' + str(accuracy) + '%')
    print('F1 score: ' + str(f1_total))
    return f1_total


# The class below is a custom callback for Keras model, where only the model weights achieving the best F1 score are saved.

# In[ ]:


class F1_callback(keras.callbacks.Callback):
    def __init__(self,X_val,Y_val,new_ecg_list,k,patience,batch_size):
        self.best_f1=0
        self.x_val=X_val
        self.y_val=Y_val
        self.best_epoch=0
        self.k=k
        self.new_ecg_list=new_ecg_list
        self.patience=patience
        self.batch_size=batch_size
    def on_epoch_end(self, epoch, logs=None):
        this_f1=make_predictions(self.model,self.x_val,self.y_val,self.new_ecg_list,self.batch_size)
        if (this_f1>self.best_f1):
            self.best_f1=this_f1
            print('F1 score improved to ',self.best_f1,', saving model...')
            self.model.save_weights('OxResNet_30s_run' + str(self.k) + '.h5')
            self.best_epoch=epoch
            print('Model saved!')
        else:
            print('F1 score was ', this_f1,'. Did not improve from ', self.best_f1)
            if (epoch>=self.best_epoch+self.patience):
                print('That is',self.patience,'epochs with no improvement. Stopping training.')
                self.model.stop_training = True
        return


# The function below trains the Keras model using weighted loss to reflect the skewed data distribution and a custom callback (see above) to only save the weights that achieve the best F1 score.
# 
# Note the class weights are specific to the Physionet AF Challenge dataset. They will need manually adjusting if using this script with an alternative dataset.

# In[ ]:


def run_training_batch(model,X_train,Y_train, X_val, Y_val, new_ecg_list,k,batch_size,epochs,patience):
    weights={} 
    for i in range(Y_train.shape[1]):
        weight_array=np.zeros(Y_train.shape[1])
        weight_array[i]=1
        this_weight=(Y_train==weight_array).all(axis=1).sum()/Y_train.shape[0]
        weights.update({i:this_weight})
    f1_callback = F1_callback(X_val,Y_val,new_ecg_list,k,patience,batch_size)
    model.fit(x=X_train, 
              y=Y_train, 
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              callbacks=[f1_callback],
              class_weight=weights)
    del X_train
    del Y_train
    return model


# The function below loads and pre-processes the data for each batch of training.

# In[ ]:


def retrain_model(ecg_list,image_y,ecg_seconds,FS,ground_truth,K,batch_size,epochs,patience,
                  classes,data_split,big_data,visualise_data_each_cycle,dropout,use_adam_optimiser,lr):
    
    data_segments_per_k=int(data_split//K)
    f1_total=0
    optimiser=Adam(lr=lr)
    if (use_adam_optimiser==False):
        optimiser=SGD(lr=lr)
    #No need to batch up the data if it will all fit into RAM:
    if (big_data==False):
        for k in range(K):
            model = ResNet_model_low_dropout(ecg_seconds*FS,classes)
            if(os.path.isfile('OxResNet_30s_run' + str(k) + '.h5')==False):
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                model.summary()
                X_train=np.zeros((1,1))
                Y_train=np.zeros((1,1))
                X_val=np.zeros((1,1))
                Y_val=np.zeros((1,1))
                new_ecg_list=[]
                first_batch_loaded=False
                for i in range(0, K*data_segments_per_k, data_segments_per_k):
                    if (i==k*data_segments_per_k):
                        X_val=np.load('training_nps/x_train_'+str(i)+'.npy')
                        Y_val=np.load('training_nps/y_train_'+str(i)+'.npy')
                        with open ('training_nps/original_record_'+str(i)+'.p', 'rb') as list_file:
                            new_ecg_list = pickle.load(list_file)
                        for z in range(i+1,i+data_segments_per_k):
                            X_val=np.concatenate((X_val,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                            Y_val=np.concatenate((Y_val,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                            with open ('training_nps/original_record_'+str(z)+'.p', 'rb') as list_file:
                                tmp_ecg_list = pickle.load(list_file)
                                new_ecg_list=new_ecg_list+tmp_ecg_list
                                del tmp_ecg_list
                    else:
                        if (first_batch_loaded==False):
                            X_train=np.load('training_nps/x_train_'+str(i)+'.npy')
                            Y_train=np.load('training_nps/y_train_'+str(i)+'.npy')
                            for z in range(i+1,i+data_segments_per_k):
                                X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                                Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                            first_batch_loaded=True
                        else:
                            X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(i)+'.npy')),axis=0)
                            Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(i)+'.npy')),axis=0)
                            for z in range(i+1,i+data_segments_per_k):
                                X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                                Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                if (visualise_data_each_cycle):
                    visualise_data(X_val,Y_val,new_ecg_list)
                X_train=np.expand_dims(X_train, axis=-1)
                X_val=np.expand_dims(X_val, axis=-1)
                print('Cycle:', k, 'of',K,'X_train:', X_train.shape, 'Y_train:', Y_train.shape, 'X_val:', X_val.shape, 'Y_val:', Y_val.shape)
                model=run_training_batch(model,X_train,Y_train,X_val,Y_val,new_ecg_list,k,batch_size,epochs,patience)
                print('Cycle ' + str(k+1) + ' of ' + str(K) + ' complete')
                del X_train
                del Y_train
                del X_val
                del Y_val
        
        for k in range(K):
            model = ResNet_model_custom_dropout(ecg_seconds*FS,classes,dropout)
            model=load_old_model(model,'OxResNet_30s_run' + str(k) + '.h5')
            sgd=SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', 
                        optimizer=optimiser, 
                        metrics=['acc'])
            model.summary()
            X_train=np.zeros((1,1))
            Y_train=np.zeros((1,1))
            X_val=np.zeros((1,1))
            Y_val=np.zeros((1,1))
            new_ecg_list=[]
            first_batch_loaded=False
            for i in range(0, K*data_segments_per_k, data_segments_per_k):
                if (i==k*data_segments_per_k):
                    X_val=np.load('training_nps/x_train_'+str(i)+'.npy')
                    Y_val=np.load('training_nps/y_train_'+str(i)+'.npy')
                    with open ('training_nps/original_record_'+str(i)+'.p', 'rb') as list_file:
                        new_ecg_list = pickle.load(list_file)
                    for z in range(i+1,i+data_segments_per_k):
                        X_val=np.concatenate((X_val,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                        Y_val=np.concatenate((Y_val,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                        with open ('training_nps/original_record_'+str(z)+'.p', 'rb') as list_file:
                            tmp_ecg_list = pickle.load(list_file)
                            new_ecg_list=new_ecg_list+tmp_ecg_list
                            del tmp_ecg_list
                else:
                    if (first_batch_loaded==False):
                        X_train=np.load('training_nps/x_train_'+str(i)+'.npy')
                        Y_train=np.load('training_nps/y_train_'+str(i)+'.npy')
                        for z in range(i+1,i+data_segments_per_k):
                            X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                            Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                        first_batch_loaded=True
                    else:
                        X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(i)+'.npy')),axis=0)
                        Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(i)+'.npy')),axis=0)
                        for z in range(i+1,i+data_segments_per_k):
                            X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                            Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
            if (visualise_data_each_cycle):
                visualise_data(X_val,Y_val,new_ecg_list)
            X_train=np.expand_dims(X_train, axis=-1)
            X_val=np.expand_dims(X_val, axis=-1)
            print('Cycle:', k, 'of',K,'X_train:', X_train.shape, 'Y_train:', Y_train.shape, 'X_val:', X_val.shape, 'Y_val:', Y_val.shape)
            model=run_training_batch(model,X_train,Y_train,X_val,Y_val,new_ecg_list,k,batch_size,epochs,patience)
            model=load_model('OxResNet_5s_run' + str(k) + '.h5')
            f1_total+=make_predictions(model,X_val,Y_val,new_ecg_list,batch_size)
            print('Cycle ' + str(k+1) + ' of ' + str(K) + ' complete')
            del X_train
            del Y_train
            del X_val
            del Y_val
            del new_ecg_list
    
    #If there is too much data, cannot load into RAM all in one go:
    
    #(Could use a generator, but I think it's as well to just train initially on one half of the data with ADAM optimiser
    #and re-train on the other half of the data with SGD + Nesterov & lr=0.001 for fine-tuning.)
    elif (big_data==True):
        try:
            training_runs_array = np.load('training_runs.npy')
        except FileNotFoundError:
            training_runs_array=np.zeros(1)
            np.save('training_runs.npy',training_runs_array)
        k=int(training_runs_array[0])
        
        #First round of training freezes base model and just trains new dense layers:
        if(os.path.isfile('OxResNet_30s_run' + str(k) + '.h5')==False):
            model = ResNet_model_low_dropout(ecg_seconds*FS,classes)
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.summary()
            X_train=np.zeros((1,1))
            Y_train=np.zeros((1,1))
            X_val=np.zeros((1,1))
            Y_val=np.zeros((1,1))
            new_ecg_list=[]
            first_batch_loaded=False
            for i in range(0, K*data_segments_per_k, data_segments_per_k):
                if (i==k*data_segments_per_k):
                    X_val=np.load('training_nps/x_train_'+str(i)+'.npy')
                    Y_val=np.load('training_nps/y_train_'+str(i)+'.npy')
                    with open ('training_nps/original_record_'+str(i)+'.p', 'rb') as list_file:
                        new_ecg_list = pickle.load(list_file)
                    for z in range(i+2,i+data_segments_per_k-1,2):
                        X_val=np.concatenate((X_val,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                        Y_val=np.concatenate((Y_val,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                        with open ('training_nps/original_record_'+str(z)+'.p', 'rb') as list_file:
                            tmp_ecg_list = pickle.load(list_file)
                            new_ecg_list=new_ecg_list+tmp_ecg_list
                            del tmp_ecg_list
                else:
                    if (first_batch_loaded==False):
                        X_train=np.load('training_nps/x_train_'+str(i)+'.npy')
                        Y_train=np.load('training_nps/y_train_'+str(i)+'.npy')
                        for z in range(i+2,i+data_segments_per_k,2):
                            X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                            Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                        first_batch_loaded=True
                    else:
                        X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(i)+'.npy')),axis=0)
                        Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(i)+'.npy')),axis=0)
                        for z in range(i+2,i+data_segments_per_k,2):
                            X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                            Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
            if (visualise_data_each_cycle):
                visualise_data(X_val,Y_val,new_ecg_list)
            X_train=np.expand_dims(X_train, axis=-1)
            X_val=np.expand_dims(X_val, axis=-1)
            print('Cycle:', k, 'of',K,'X_train:', X_train.shape, 'Y_train:', Y_train.shape, 'X_val:', X_val.shape, 'Y_val:', Y_val.shape)
            model=run_training_batch(model,X_train,Y_train,X_val,Y_val,new_ecg_list,k,batch_size,epochs,patience)
            print('Cycle ' + str(k+1) + ' of ' + str(K) + ' complete')
            k+=1
            if (k<K):
                training_runs_array[0]=k
                np.save('training_runs.npy',training_runs_array)
                restartkernel()
            else:
                k=0
                training_runs_array[0]=k
                np.save('training_runs.npy',training_runs_array)
                restartkernel()
        
        #After initial training, base model is unfrozen and the whole model is retrained:
        try:
            f1_results_array = np.load('f1_results_array.npy')
        except FileNotFoundError:
            f1_results_array=np.zeros(K)
            np.save('f1_results_array.npy',f1_results_array)
        model = ResNet_model_custom_dropout(ecg_seconds*FS,classes,dropout)
        model=load_old_model(model,'OxResNet_30s_run' + str(k) + '.h5')
        model.compile(optimizer=optimiser,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        model.summary()
        X_train=np.zeros((1,1))
        Y_train=np.zeros((1,1))
        X_val=np.zeros((1,1))
        Y_val=np.zeros((1,1))
        new_ecg_list=[]
        first_batch_loaded=False
        for i in range(1, K*data_segments_per_k, data_segments_per_k):
            if (i==k*data_segments_per_k+1):
                X_val=np.load('training_nps/x_train_'+str(i)+'.npy')
                Y_val=np.load('training_nps/y_train_'+str(i)+'.npy')
                with open ('training_nps/original_record_'+str(i)+'.p', 'rb') as list_file:
                    new_ecg_list = pickle.load(list_file)
                for z in range(i+2,i+data_segments_per_k-1,2):
                    X_val=np.concatenate((X_val,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                    Y_val=np.concatenate((Y_val,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                    with open ('training_nps/original_record_'+str(z)+'.p', 'rb') as list_file:
                        tmp_ecg_list = pickle.load(list_file)
                        new_ecg_list=new_ecg_list+tmp_ecg_list
                        del tmp_ecg_list
            else:
                if (first_batch_loaded==False):
                    X_train=np.load('training_nps/x_train_'+str(i)+'.npy')
                    Y_train=np.load('training_nps/y_train_'+str(i)+'.npy')
                    for z in range(i,i+data_segments_per_k-1,2):
                        X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                        Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
                    first_batch_loaded=True
                else:
                    X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(i)+'.npy')),axis=0)
                    Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(i)+'.npy')),axis=0)
                    for z in range(i+2,i+data_segments_per_k-1,2):
                        X_train=np.concatenate((X_train,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
                        Y_train=np.concatenate((Y_train,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
        if (visualise_data_each_cycle):
            visualise_data(X_val,Y_val,new_ecg_list)
        X_train=np.expand_dims(X_train, axis=-1)
        X_val=np.expand_dims(X_val, axis=-1)
        print('Cycle:', k+1, 'of',K,'X_train:', X_train.shape, 'Y_train:', Y_train.shape, 'X_val:', X_val.shape, 'Y_val:', Y_val.shape)
        model=run_training_batch(model,X_train,Y_train,X_val,Y_val,new_ecg_list,k,batch_size,epochs,patience)
        model.load_weights('OxResNet_30s_run' + str(k) + '.h5')

        del X_train
        del Y_train

        #Load full validation data:
		X_val=np.load('training_nps/x_train_'+str(k*data_segments_per_k)+'.npy')
		Y_val=np.load('training_nps/y_train_'+str(k*data_segments_per_k)+'.npy')
		with open ('training_nps/original_record_'+str(k*data_segments_per_k)+'.p', 'rb') as list_file:
			new_ecg_list = pickle.load(list_file)
		for z in range((k*data_segments_per_k)+1,(k+1)*data_segments_per_k):
				X_val=np.concatenate((X_val,np.load('training_nps/x_train_'+str(z)+'.npy')),axis=0)
				Y_val=np.concatenate((Y_val,np.load('training_nps/y_train_'+str(z)+'.npy')),axis=0)
				with open ('training_nps/original_record_'+str(z)+'.p', 'rb') as list_file:
					tmp_ecg_list = pickle.load(list_file)
					new_ecg_list=new_ecg_list+tmp_ecg_list
					del tmp_ecg_list
		X_val=np.expand_dims(X_val, axis=-1)


        f1_total=make_predictions(model,X_val,Y_val,new_ecg_list,batch_size)
        print('Cycle ' + str(k+1) + ' of ' + str(K) + ' complete')
        
        f1_results_array[k]=f1_total
        k+=1
        training_runs_array[0]=k
        np.save('f1_results_array.npy',f1_results_array)
        np.save('training_runs.npy',training_runs_array)
        if (k<K):
            restartkernel()
        else:
            del X_val
            del Y_val
            del new_ecg_list
            del model
            
    #Final lines are common to all above funcitons (i.e. regardless of big_data variable):
    results_text=''
    for f in f1_results_array:
        results_text+=str(f)+'\n'
    f1_total=sum(f1_results_array)/K
    print('Completed ',K,'-fold cross-validation. F1 score: ',f1_total)
    text_file = open("F1_score_5-fold-CV.txt", "w")
    results_text+="\nF1_score after "+ str(k) +"-fold validation: " + "{:.2f}".format(f1_total)
    text_file.write(results_text)
    text_file.close()


# The function below produces a random sample of five ECGs in human-readable forms.

# In[ ]:


def visualise_data(X,Y,new_ecg_list):
    for i in range(5):
        rand=randint(0,X.shape[0])
        fig, ax = plt.subplots()
        split_str=new_ecg_list[rand].split('.')[1]
        split_str=split_str.split('/')[-1]
        split_str=split_str.split('_')[0]
        title=split_str + ': ' + one_hot_to_label(Y[rand,:])
        ax.plot(X[rand,0:1000],'k')
        ax.set_title(title)
        plt.show()


# In[ ]:


def ResNet_model_custom_dropout(WINDOW_SIZE,OUTPUT_CLASS,dropout):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    resnet_model=load_model('ResNet_30s_34lay_16conv.hdf5')
    for l in resnet_model.layers:
        l.trainable=False
    resnet_model.layers.pop()
    inp = resnet_model.input
    out =resnet_model.layers[-1].output
    model2 = Model(inp, out)
    model=Sequential()
    model.add(model2)
    model.add(Dense(512,activation='relu',name='new_layer_1'))
    model.add(Dropout(dropout,name='new_layer_2'))
    model.add(Dense(512,activation='relu',name='new_layer_3'))
    model.add(Dropout(dropout,name='new_layer_4'))
    model.add(Dense(4,activation='softmax',name='new_layer_5'))
    return model


# The following two functions take the pretrained model from the University of Oxford and tweak the architecture a little.

# In[ ]:


def ResNet_model_low_dropout(WINDOW_SIZE,OUTPUT_CLASS):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    resnet_model=load_model('ResNet_30s_34lay_16conv.hdf5')
    for l in resnet_model.layers:
        l.trainable=False
    resnet_model.layers.pop()
    inp = resnet_model.input
    out =resnet_model.layers[-1].output
    model2 = Model(inp, out)
    model=Sequential()
    model.add(model2)
    model.add(Dense(512,activation='relu',name='new_layer_1'))
    model.add(Dropout(0,name='new_layer_2'))
    model.add(Dense(512,activation='relu',name='new_layer_3'))
    model.add(Dropout(0,name='new_layer_4'))
    model.add(Dense(4,activation='softmax',name='new_layer_5'))
    return model


# In[ ]:


def load_old_model(old_model,weights_path):
    try:
        old_model.load_weights(weights_path)
    except ValueError:
        pass
    resnet_model=load_model('ResNet_30s_34lay_16conv.hdf5')
    resnet_model.layers.pop()
    inp = resnet_model.input
    out =resnet_model.layers[-1].output
    model2 = Model(inp, out)
    model=Sequential()
    model.add(model2)
    for l in old_model.layers[-5:]:
        model.add(l)
    try:
        model.load_weights(weights_path)
    except ValueError:
        pass
    return model


# The function below is an inelegant workaround to dump the cache. See the following function for clarification.

# In[ ]:


def restartkernel() :
    #from IPython.display import display_html
    #display_html("<script>Jupyter.notebook.kernel.restart();restartTime = 5000;setTimeout(function(){ Jupyter.notebook.execute_all_cells(); }, restartTime);</script>",raw=True)
    main()


# This is a real faff, but due to a memory leak in MatPlotLib, the RAM clogged up when trying to create and write a large number of images to disk. Restarting the iPython notebook dumped the memory and sorts this out.
# 
# The number of cycles is written to disk and loaded up every time the script is restarted. In this particular iteration of the script, the machine used during our study can handle batches of 250 records (bearing in mind each record = approx 10-20 ECG images) and we're dealing with just under 10,000 ECG records, so 10,000/250 = 40 cycles.
# 
# When we have run this code on newer machines as a straight Python script (i.e. outside Jupyter), the issue has not occurred. To do this, simply save this script as .py file and replace the restartkernel() function thus:
# ```python
# def restartkernel():
#     main()
# ```

# In[ ]:


def prep_main(image_y,ecg_seconds,FS,directory,sliding_window_seconds,max_seconds_of_padding):   
    records_per_batch=250
    total_cycles=40
    
    try:
        script_runs_array = np.load('script_runs.npy')
    except FileNotFoundError:
        script_runs_array=np.zeros(1)
        np.save('script_runs.npy',script_runs_array)
    script_runs=int(script_runs_array[0])
    
    if (script_runs<total_cycles):
        print('Running batch',script_runs+1,'of,',total_cycles)
        [ecg_list,ground_truth,original_ecg_list]=create_ecg_images(image_y,ecg_seconds,FS,'training2017',sliding_window_seconds,max_seconds_of_padding,script_runs,records_per_batch)
        script_runs+=1
        script_runs_array[0]=script_runs
        np.save('script_runs.npy',script_runs_array)
        restartkernel()
    else:
         [ecg_list,ground_truth,original_ecg_list]=load_lists()
            
    return [ecg_list,ground_truth,original_ecg_list]


# The function below executes the functions in order. Adjustable variables can be set here if the code is being used with another dataset.

# In[ ]:


def main():
    image_y=150 # height of generated ECG images in px, see manually_read_image function for explanation of why 150 chosen
    ecg_seconds=30 # set at 30 to capitalise on pre-trained model from Oxford paper
    FS=300 # frames / second (ECG resolution in Hz within source data)
    data_split=20 # to ensure the data is processed in chunks that fit in RAM (20 works on my machine)
    big_data=True # if processed data-set (i.e. number of images x (FS * ecg_seconds)-dimensional vectors) too big for RAM
    k=5 # for k-fold validation
    epochs=1000 # for ML model
    batch_size=64 # for ML model
    patience=10 # for ML model
    classes=4 # for ML model
    sliding_window_seconds=1 # create more training examples if records are > ecg_seconds
    max_seconds_of_padding=10 # pad records so model can cope with records < ecg_seconds
    F1_list=np.zeros((1,12)) # corresponding to: A N O P a n o p Aa Nn Oo Pp
    #(where A=AF, N=NSR, O=other, P=too noisy to interpret. Criteria from Physionet 2017 AF Challenge.)
    visualise_data=False # prints a selection of data before each training cycle
    
    #Note: ML model will always run once with a frozen base model and Adam optimiser with dropout=0 for (new) dense
    #layers (2 x 512 with ReLU -> output with )
    dropout=0.3 # for final, dense layers of model
    use_adam_optimiser=True # if false will use SGD with Nesterov
    lr=0.001 # learning rate
    
    #Creates lists of images, labels and original ECGs:
    ecg_list,ground_truth,original_ecg_list=prep_main(image_y,ecg_seconds,FS,classes,
                                                      sliding_window_seconds,max_seconds_of_padding)
    
    #Generates, imports and processes images in batches, saves resulting ndarrays to disk:
    create_x_batches(ecg_list,image_y,ecg_seconds,FS,ground_truth,data_split,original_ecg_list)
    
    #Creates and trains the model then prints F1 score:
    retrain_model(ecg_list,image_y,ecg_seconds,FS,ground_truth,k,batch_size,epochs,patience,
                  classes,data_split,big_data,visualise_data,dropout,use_adam_optimiser,lr)


# If this code is run as a straight Python script (i.e. not in Jupyter), this is the part that will go within the
# ```python
# if __name__=='__main__':
# ```
# clause:

# In[ ]:


main()


# In[ ]:




