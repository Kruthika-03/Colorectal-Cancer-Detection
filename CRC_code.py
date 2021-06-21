import numpy as np 
import pandas as pd 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix

data = pd.read_csv("hmnist_64_64_L.csv")

data.head()  

Y = data["label"]
data.drop(["label"],axis=1, inplace=True)
X = data

X = X / 255.0     # scaling by hand since we know the max value

img = X.iloc[75].to_numpy()
img = img.reshape(64,64)
plt.imshow(img)
plt.suptitle("An example of image on the dataset")
plt.show()

# Reshaping
X = X.values.reshape(-1,64,64,1)       # shaping for the Keras
Y = Y.values

# Label Encoding 
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding for better results
Y = to_categorical(Y)

# Splitting train and test
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)

print("x_train.shape: ",x_train.shape)
print("x_val.shape: ",x_val.shape)
print("y_train.shape: ",y_train.shape)
print("y_val.shape: ",y_val.shape)

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (64,64,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))          
model.add(Dense(64,activation = "relu"))
model.add(Dense(32,activation = "relu"))

model.add(Dense(8, activation = "softmax"))

model.summary()
model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
        rotation_range=0.5, 
        zoom_range = 0.5, 
        width_shift_range=0.5,  
        height_shift_range=0.5, 
        horizontal_flip=True, 
        vertical_flip=True)

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=200),
                              epochs = 20, validation_data = (x_val,y_val), steps_per_epoch=500)

# Predict the values from the validation dataset
Y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="summer_r", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.plot(history.history['loss'],color='b',label='Training Loss')
plt.plot(history.history['val_loss'],color='r',label='Validation loss')
plt.show()
plt.plot(history.history['acc'],color='b',label='Training accuracy')
plt.plot(history.history['val_acc'], color='r',label='Validation Accuracy')
plt.show()
