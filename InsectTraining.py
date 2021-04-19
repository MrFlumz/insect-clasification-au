import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
import glob
import numpy as np
import cv2
import os
instances = []
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import glob

from Resizing import resizeImg
imageSize = 128
resizeImg(imageSize, imgdir = "data/Train/TrainImages/", overwrite = False)
resizeImg(imageSize, imgdir = "data/Validation/ValidationImages/", overwrite = False)

dir = "data/Train/TrainImages/" + str(imageSize) + "/"
#print(len(os.listdir(dir)))
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_train = np.array([np.array(Image.open(i).convert('L')) for i in fname]) # The L means that is just stores the Luminance

plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.show()


dir = "data/Validation/ValidationImages/" + str(imageSize)  + "/"
print(len(os.listdir(dir)))
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_validation = np.array([np.array(Image.open(i).convert('L')) for i in fname]) # The L means that is just stores the Luminance

nr, x, y = x_train.shape
print("x_train shape:" + str(x_train.shape))
X_train = x_train.reshape(nr,x,y,1)

nr, x, y = x_validation.shape
print("x_validation shape:" + str(x_validation.shape))
X_validation = x_validation.reshape(nr,x,y,1)

Y_train = np.load("data/Train/trainLbls.npy")
Y_validation = np.load("data/Validation/valVectors.npy")
from sklearn.model_selection import train_test_split
#X_train, X_validation, Y_train, Y_validation = train_test_split( X_train, Y_train, test_size=0.25, random_state=42)

from tensorflow.keras.utils import to_categorical
#Y_train = to_categorical(Y_train)
#Y_validation = to_categorical(Y_validation)
#%%


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_validation = scaler.transform(X_validation.reshape(-1, X_validation.shape[-1])).reshape(X_validation.shape)

Y_train = Y_train.astype(int) - 1 # labels from 0-29
Y_validation = Y_validation.astype(int) - 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
#add model layers
#model.add(Conv2D(64, kernel_size=3, activation='relu',activity_regularizer=l2(0.005)))
from keras import backend
def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return fbeta_score



model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(29, activation='sigmoid'))

#compile model using accuracy to measure model performance
from sklearn.metrics import fbeta_score


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
#%%
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, Y_train, batch_size=32), validation_data=(X_validation, Y_validation), epochs=50, steps_per_epoch=len(X_train) / 32)
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



