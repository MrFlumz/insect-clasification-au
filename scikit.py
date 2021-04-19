import numpy as np
from sklearn import linear_model
from numpy import genfromtxt
import keras
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#%%
print("loading testdata")
TrainFile = "data/Train/trainVectors.txt"
trainData = np.loadtxt(TrainFile, delimiter=' ')

TrainLabelFile = "data/Train/trainLbls.txt"
trainLab = np.loadtxt(TrainLabelFile, delimiter=' ')

print("loading validation data")
ValidationFile = "data/Validation/valVectors.txt"
ValData = np.loadtxt(ValidationFile, delimiter=' ')

ValidationLabelFile = "data/Validation/valLbls.txt"
ValLab = np.loadtxt(ValidationLabelFile, delimiter=' ')

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
reg = OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0005, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=2, class_weight='balanced', verbose=0, random_state=1, max_iter=2000))
#reg = linear_model.SGDClassifier(learning_rate="adaptive", eta0=0.0005, max_iter=2000)
#reg = linear_model.PassiveAggressiveClassifier()
# reg = linear_model.LogisticRegression()
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#reg = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(128,64),activation="relu",  alpha=0.5e-5, random_state=1, max_iter=1000)
#sc = StandardScaler()
sc.fit(trainData.T)
trainDataS = sc.transform(trainData.T)
ValDataS = sc.transform(ValData.T)
print("fitting data")
reg.fit(trainDataS, trainLab)  # .T flattens 2d to 1d
print("model score"+str(reg.score(trainData, trainLab)))


# prediction
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
Pred = reg.predict(ValData)
# mcm = multilabel_confusion_matrix(ValLab, Pred)
# print(mcm)
score = precision_score(ValLab, Pred, average='weighted')
print("Training score is : " + str(score))
