import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import optimizers


def nvidia_architecture():
    # https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    # I'm sure about kernel sizes and strides, not about the padding, it should be same for the first 3 layers
    
    drop_prob = 0.0
    # initializier = 'lecun_uniform'
    initializier = 'glorot_uniform'
    
    height = 160
    width = 320
    channels = 3
    
    model = Sequential()
    
    model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(height, width, channels)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))  
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Flatten())
    
    model.add(Dense(100, init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Dense(50, init=initializier))
    model.add(ELU())
    #model.add(Dropout(drop_prob))
    
    model.add(Dense(10, init=initializier))
    model.add(ELU())
    
    model.add(Dense(1, init=initializier))
    
    return model


def  lenet_architecture():
    model = Sequential()
    
    model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(height, width, channels)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
    return model


### Parameters
data_folder = 'mydata/'
skip_first = True
epochs = 20
batch_size = 128
use_lateral_images = False
flip_dataset = False
model_type = 'nvidia'
learning_rate = 1e-03
train_augmentation = 1 * (3 if use_lateral_images else 1) * (2 if flip_dataset else 1)


### Import and split data log
samples = []
with open(data_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    first = True
    for row in reader:
        if not first:
            samples.append(row)
        else:
            first = False

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


### Generator function
def generator(samples, batch_size=32, mode='train'):
    num_samples = len(samples)
    while True:
    
        # Shuffle data set for every batch
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Read steering angle and compute adjusted steering measurements for the side camera images
                
                steering_center = float(batch_sample[3])
                correction = 0.5 # tuning: 1.0 is the one showing the best results, I don't understand why 0.9 or 1.1 are already incredibly bad!
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # Read in images from center, left and right cameras. Images are loaded as BGR as default, I change the color space to YUV.
                path = data_folder + 'IMG/'
                img_center = cv2.cvtColor(cv2.imread(path + batch_sample[0].split('/')[-1]), cv2.COLOR_BGR2YUV)
                img_left   = cv2.cvtColor(cv2.imread(path + batch_sample[1].split('/')[-1]), cv2.COLOR_BGR2YUV)
                img_right  = cv2.cvtColor(cv2.imread(path + batch_sample[2].split('/')[-1]), cv2.COLOR_BGR2YUV)
                
                # Add images and angles to dataset
                if use_lateral_images and (mode == 'train'):
                    images_to_append = [img_center, img_left, img_right]
                    angles_to_append = [steering_center, steering_left, steering_right]
                    
                else:
                    images_to_append = [img_center]
                    angles_to_append = [steering_center]
                
                # Add a flipped version of each image
                if flip_dataset and (mode == 'train'):
                    for image in images_to_append:
                        images_to_append.append(cv2.flip(image, 1))
                    for angle in angles_to_append:
                        angles_to_append.append(-1. * angle)
                
                # Append the results to the output list
                images.extend(images_to_append)
                angles.extend(angles_to_append)
                #print(images_to_append)
                #print(angles_to_append)
            
            # Convert to numpy array and return
            X_train = np.array(images)
            y_train = np.array(angles)
                        
            yield sklearn.utils.shuffle(X_train, y_train)


### Training
train_generator = generator(train_samples, batch_size=batch_size, mode='train')
validation_generator = generator(validation_samples, batch_size=batch_size, mode='valid')

model = nvidia_architecture()
        
check_point = ModelCheckpoint('./checkpoints/model-e{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

model.compile(loss='mse', optimizer='adam')

"""
model.compile(loss='mse',
              optimizer=optimizers.Adam(lr=learning_rate))
"""

model.fit_generator(train_generator,
                    samples_per_epoch= len(train_samples) * train_augmentation,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=epochs,
                    verbose=1,
                    callbacks=[early_stop, check_point])

                    
### Saving
model.save('model.h5')


### Show some stats                    
f, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_title('model mean squared error loss')
ax.set_ylabel('mean squared error loss')
ax.set_xlabel('epoch')
ax.grid(True)
plt.legend(['training set', 'validation set'], loc='upper right')
figname = 'model_' + str(use_lateral_images) + '_' + str(flip_dataset) + '_' + model_type + '_' + str(epochs) + '_' + str(batch_size) + '.png'
f.savefig(figname, bbox_inches='tight')

