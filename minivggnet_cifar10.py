import warnings
warnings.simplefilter('ignore')
from hvdev.nn.cnn import MiniVggNet
from keras.optimizers import SGD
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
import numpy as np 

print('[INFO] loading dataset...')
(trainX , trainY), (testX , testY) = cifar10.load_data()

print('[INFO] preprocessing data...')
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

classesName = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print('[INFO] compiling model...')
model = MiniVggNet().build(height = 32, width = 32 , depth  = 3, classes = 10)
sgd = SGD(lr = 0.01, decay = 0.01/40, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd , metrics = ['accuracy'])

print('[INFO] training model...')
H = model.fit(trainX , trainY , validation_data = (testX , testY), epochs = 40 , 
    batch_size = 64, verbose = 1)

print('[INFO] Evaluating network')
predictions = model.predict(testX , batch_size = 64).argmax(axis = 1)
testY = testY.argmax(axis = 1)

print(classification_report(testY , predictions, target_names = classesName))

print('[INFO] ploting model...')
plt.style.use('ggplot')
plt.plot(np.arange(0 , 40), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0 , 40), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0 , 40), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0 , 40), H.history['val_acc'], label = 'val_acc')
plt.title('Training/Validation Loss/Accuracy')
plt.xlabel('#epochs')
plt.ylabel('loss')
plt.legend()
plt.show()