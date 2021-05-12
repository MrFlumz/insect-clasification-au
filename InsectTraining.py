#%%

import numpy as np
import os
instances = []
# https://pillow.readthedocs.io/en/stable/index.html
from PIL import Image
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

from Resizing import resizeImg
imageSize = 128
resizeImg(imageSize, imgdir = "data/Train/TrainImages/", overwrite = False)
resizeImg(imageSize, imgdir = "data/Validation/ValidationImages/", overwrite = False)
resizeImg(imageSize, imgdir = "data/Test/TestImages/", overwrite = False)
#%%
dir = "data/Train/TrainImages/" + str(imageSize) + "/"
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_train = np.array([np.array(Image.open(i)) for i in fname]) # The L means that is just stores the Luminance

plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.show()


dir = "data/Validation/ValidationImages/" + str(imageSize)  + "/"
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_validation = np.array([np.array(Image.open(i)) for i in fname]) # The L means that is just stores the Luminance

dir = "data/Test/TestImages/" + str(imageSize)  + "/"
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_test = np.array([np.array(Image.open(i)) for i in fname]) # The L means that is just stores the Luminance

nr, x, y, c = x_train.shape
print("x_train shape:" + str(x_train.shape))
X_train = x_train.reshape(nr,x,y,c)

nr, x, y, c = x_validation.shape
print("x_validation shape:" + str(x_validation.shape))
X_validation = x_validation.reshape(nr,x,y,c)

nr, x, y, c = x_test.shape
print("x_test shape:" + str(x_test.shape))
X_test = x_test.reshape(nr,x,y,c)

#%% load test labels
Y_train = np.load("data/Train/trainLbls.npy")
Y_validation = np.load("data/Validation/valVectors.npy")
Y_train = Y_train.astype(int) - 1 # labels from 0-29
Y_validation = Y_validation.astype(int) - 1

# Normalize data - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
# https://stackoverflow.com/a/59601298
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_validation = scaler.transform(X_validation.reshape(-1, X_validation.shape[-1])).reshape(X_validation.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


#%%
from sklearn.utils import class_weight
Class_Weights = class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train)
Class_Weights = {i : Class_Weights[i] for i in range(np.max(Y_train)+1)}
#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
model = Sequential()
from tensorflow.keras.regularizers import l2
RegulazationValue = 0.0001
model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform', padding='same', ))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform', padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform'))
model.add(Dense(29, kernel_regularizer=l2(RegulazationValue), activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='Adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')

print(model.summary())
#%% https://keras.io/api/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
rotation_range = 20,
width_shift_range = 0.2,
height_shift_range = 0.2,
zoom_range=0.2,
shear_range = 0.2,
horizontal_flip = True)
datagen.fit(X_train)

#%%
history = model.fit(datagen.flow(X_train, Y_train), validation_data=(X_validation, Y_validation), epochs=80, steps_per_epoch=len(X_train) / 64, shuffle=True, class_weight=Class_Weights)
#%%
p = model.predict(X_validation)
np.set_printoptions(edgeitems=20)
np.set_printoptions(linewidth=200)
print(np.argmax((model.predict(X_validation)), 1))
print(Y_validation.astype(int))

from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
Pred = np.argmax((model.predict(X_validation)), 1)
# mcm = multilabel_confusion_matrix(ValLab, Pred)
# print(mcm)
score = precision_score(Y_validation, Pred, average='weighted')
print("Training score is : " + str(score))


#%%
# Save testresults in format that matches what kaggle wants
Y_test = np.argmax((model.predict(X_test)), 1)
Y_test_output = np.empty((np.size(Y_test),2),dtype=int)
for i in range(0, np.size(Y_test)):
    Y_test_output[i,:] = np.array([(i+1), 1+Y_test[i].astype(int)])
np.savetxt("Testresults.csv", Y_test_output.astype(int), fmt='%i', delimiter=",",header="ID,Label",comments='')




print(history.history.keys())
# plot accuracy and loss
# https://stackoverflow.com/a/56807595
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

