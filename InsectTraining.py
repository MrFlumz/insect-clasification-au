#%%

import numpy as np
import os
instances = []
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from Resizing import resizeImg
imageSize = 128
resizeImg(imageSize, imgdir = "data/Train/TrainImages/", overwrite = False)
resizeImg(imageSize, imgdir = "data/Validation/ValidationImages/", overwrite = False)
resizeImg(imageSize, imgdir = "data/Test/TestImages/", overwrite = False)
#%%
dir = "data/Train/TrainImages/" + str(imageSize) + "/"
#print(len(os.listdir(dir)))
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_train = np.array([np.array(Image.open(i)) for i in fname]) # The L means that is just stores the Luminance

plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.show()


dir = "data/Validation/ValidationImages/" + str(imageSize)  + "/"
print(len(os.listdir(dir)))
fname = []
for x in range(1, 1+len(os.listdir(dir))):
    fname.append( dir + "Image"+str(x)+".jpg")
x_validation = np.array([np.array(Image.open(i)) for i in fname]) # The L means that is just stores the Luminance

dir = "data/Test/TestImages/" + str(imageSize)  + "/"
print(len(os.listdir(dir)))
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

#%%
Y_train = np.load("data/Train/trainLbls.npy")
Y_validation = np.load("data/Validation/valVectors.npy")
Y_train = Y_train.astype(int) - 1 # labels from 0-29
Y_validation = Y_validation.astype(int) - 1
from sklearn.model_selection import train_test_split
#X_train, X_validation, Y_train, Y_validation = train_test_split( X_train, Y_train, test_size=0.25, random_state=42)

from tensorflow.keras.utils import to_categorical
#Y_train = to_categorical(Y_train)
#Y_validation = to_categorical(Y_validation)
#%%
#ocurrences = []
#for x in range(1, np.amax(Y_train,0).astype(int)+1):
#    #print("Counting occurences of "+str(x))
#    ocurrences.append(np.count_nonzero(Y_train == x))
#largestclass = np.amax(ocurrences,0)
#smallestclass = np.amin(ocurrences,0)
#scalefactor = 0
#classWeights = (1 + (smallestclass) / (largestclass+scalefactor)) - (ocurrences/(largestclass+scalefactor))


scaler = StandardScaler()
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
#model = Sequential()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
#add model layers
#model.add(Conv2D(64, kernel_size=3, activation='relu',activity_regularizer=l2(0.005)))
from keras import backend
RegulazationValue = 0.001
# model.add(Dense(29, kernel_regularizer=l2(RegulazationValue), activation='softmax'))
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, GlobalAvgPool2D, GaussianNoise
input_tensor = Input(shape=(128, 128, 3))
model = Sequential()
model.add( InceptionV3(input_shape=(imageSize, imageSize, 3), weights='imagenet', include_top=False))

regularizer = l2(RegulazationValue)

for layer in model.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
          setattr(layer, attr, regularizer)

#for layer in model.layers[:]:
#    layer.trainable = False
model.add(Dropout(0.6))
model.add(GlobalAvgPool2D())
model.add(Dense(4096, activation='relu',kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform'))
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu',kernel_regularizer=l2(RegulazationValue), kernel_initializer='he_uniform'))
model.add(Dropout(0.6))
model.add(Dense(29, activation='softmax',kernel_regularizer=l2(RegulazationValue)))

import tensorflow.keras.optimizers as optimizers
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'], loss='sparse_categorical_crossentropy')

#compile model using accuracy to measure model performance
from sklearn.metrics import fbeta_score

print(model.summary())
#%%
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
history = model.fit(datagen.flow(X_train, Y_train, batch_size=64), validation_data=(X_validation, Y_validation), epochs=100, steps_per_epoch=len(X_train) / 64, shuffle=True, class_weight=Class_Weights)
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
Y_test = np.argmax((model.predict(X_test)), 1)
Y_test_output = np.empty((np.size(Y_test),2),dtype=int)
#Y_test_output.append("ID,Label")
for i in range(0, np.size(Y_test)):
    Y_test_output[i,:] = np.array([(i+1), 1+Y_test[i].astype(int)])
    #Y_test_output.append(str(label)+","+str(Y_test[label]))

np.savetxt("Testresults.csv", Y_test_output.astype(int), fmt='%i', delimiter=",",header="ID,Label",comments='')

#%%
#from datetime import datetime
#now = datetime.now()
#model.save("Model_"+now.strftime("%d/%m/%Y_%H:%M:%S"))
#%%
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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


#%%
ocurrences = []
for x in range(1, np.amax(Pred,0).astype(int)+1):
    #print("Counting occurences of "+str(x))
    ocurrences.append(np.count_nonzero(Pred == x))
ocurrencespred = []
for x in range(1, np.amax(Y_validation,0).astype(int)+1):
    #print("Counting occurences of "+str(x))
    ocurrencespred.append(np.count_nonzero(Y_validation == x))

products = []
for num1, num2 in zip(ocurrences, ocurrencespred):
	products.append(num1 / num2)
print(ocurrences)
print(ocurrencespred)
