from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import h5py
data_path = "./data.mat"
data_raw = loadmat(data_path)

train_img = data_raw["train_img"]
test_img = data_raw["test_img"]
Y_train = data_raw["train_lbl"]
X_train = np.reshape(train_img,(len(train_img),28,28))
X_test = np.reshape(test_img, (len(test_img),28,28))
X_valid = X_train[40000:]
Y_valid = Y_train[40000:]

np.random.seed(25)
print("X_train original shape", X_train.shape)
print("Y_train original shape", Y_train.shape)
print("X_test original shape",  X_test.shape)
print("X_valid original shape", X_valid.shape)
print("Y_valid original shape", Y_valid.shape)

#pattern 1 generation
pattern1_X_train = []
pattern1_Y_train = []
pattern1_num = 5
for i in range(len(X_train)):
    print(i)
    img = X_train[i]
    for j in range(pattern1_num):
        img_new = copy.deepcopy(img)
        rand_x = randint(4,14)
        rand_y = randint(4,14)
        for x in range(10):
            for y in range(10):
                prob = randint(0,1)
                img_new[rand_x+x,rand_y+y] = 255*prob
        pattern1_X_train.append(img_new)
        pattern1_Y_train.append(Y_train[i])
pattern1_X_train = np.array(pattern1_X_train)
pattern1_Y_train = np.array(pattern1_Y_train)
print(pattern1_X_train.shape)
# ###########################

#pattern 2 generation
pattern2_X_train = []
pattern2_Y_train = []
pattern2_num = 5
for i in range(len(X_train)):
    print(i)
    img = X_train[i]
    for k in range(pattern2_num):
        img_new = copy.deepcopy(img)
        # rand_x = randint(4,23)
        # rand_y = randint(4,23)
        for x in range(4,24):
            for y in range(4,24):
                # prob = randint(0,1)
                noise=np.random.normal(0,100)
                noise=int(noise)
                sum= img_new[x,y]+noise
                if sum>255:
                    img_new[x,y]=255
                elif  sum<0:
                    img_new[x,y]=0
                else:
                    img_new[x,y]=sum
        pattern2_X_train.append(img_new)
        pattern2_Y_train.append(Y_train[i])
pattern2_X_train = np.array(pattern2_X_train)
print(pattern2_X_train.shape)
# ###############################################

# validation set preparation
#pattern 1 generation
pattern1_X_valid = []
pattern1_Y_valid = []
pattern1_num = 2
for i in range(len(X_train)):
    print(i)
    img = X_train[i]
    for j in range(pattern1_num):
        img_new = copy.deepcopy(img)
        rand_x = randint(4,14)
        rand_y = randint(4,14)
        for x in range(10):
            for y in range(10):
                prob = randint(0,1)
                img_new[rand_x+x,rand_y+y] = 255*prob
        pattern1_X_valid.append(img_new)
        pattern1_Y_valid.append(Y_train[i])
pattern1_X_valid = np.array(pattern1_X_valid)
pattern1_Y_valid = np.array(pattern1_Y_valid)
print(pattern1_X_valid.shape)
# ###########################

#pattern 2 generation
pattern2_X_valid = []
pattern2_Y_valid = []
pattern2_num = 1
for i in range(len(X_train)):
    print(i)
    img = X_train[i]
    for k in range(pattern2_num):
        img_new = copy.deepcopy(img)
        # rand_x = randint(4,23)
        # rand_y = randint(4,23)
        for x in range(4,24):
            for y in range(4,24):
                # prob = randint(0,1)
                noise=np.random.normal(0,100)
                noise=int(noise)
                sum= img_new[x,y]+noise
                if sum>255:
                    img_new[x,y]=255
                elif  sum<0:
                    img_new[x,y]=0
                else:
                    img_new[x,y]=sum
        pattern2_X_valid.append(img_new)
        pattern2_Y_valid.append(Y_train[i])
pattern2_X_valid = np.array(pattern2_X_valid)
print(pattern2_X_valid.shape)
# ###############################################

X_valid = np.concatenate((X_train,pattern1_X_valid,pattern2_X_valid))
Y_valid = np.concatenate((Y_train,pattern1_Y_valid,pattern2_Y_valid))


X_train = np.concatenate((X_train,pattern1_X_train,pattern2_X_train))
Y_train = np.concatenate((Y_train,pattern1_Y_train,pattern2_Y_train))

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

number_of_classes = 10

Y_train = np_utils.to_categorical(Y_train, number_of_classes)
Y_valid = np_utils.to_categorical(Y_valid, number_of_classes)
X_train/=255
X_valid/=255
X_test/=255


# model structure start
model = Sequential()
model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(1,1), padding='same'))
model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(1024))
model.add(Activation('relu'))
#model.add(Dense(256))
#model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.5))
model.add(Dense(10))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
model.summary()

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.06, shear_range=0.3,
                         height_shift_range=0.06, zoom_range=0.06)

#gen = ImageDataGenerator()
test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_valid, Y_valid, batch_size=64)

filepath="./adam/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(train_generator, steps_per_epoch=len(X_train)//64, epochs=30,callbacks=callbacks_list,validation_data=test_generator, validation_steps=len(X_valid)//64)

# save and reload model
model.save('1_5_5_epochs_40_valid_structure_sgd_check.h5')

# prediction
score  =  model.evaluate(X_valid, Y_valid)
print()
print('valid accuracy: ', score[1])

# output the file
prediction = model.predict_classes(X_test)
prediction = list(prediction)
ID = list(range(1,20001))
sub = pd.DataFrame({'ID': ID, 'Prediction': prediction})
sub.to_csv('./1_5_5_epochs_40_valid_structure_sgd_check.csv', index=False)
