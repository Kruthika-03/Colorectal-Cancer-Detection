# Colorectal-Cancer-Detection
Colorectal Cancer Detection Using Deep Learning

1. Step 1 Import NumPy, TensorFlow and other required libraries
1. Step 2 Read the data from a csv file
Step 3 Check the shape and datatype
Step 4 Reshape
Step 5 Split the data into training and testing dataset
Step 6 Using to_categorical in keras to get a binary class matrix from vector class
Step 7 Define the model architecture
Step 8 Compile the model using optimizer
Step 9 Increasing the dataset using ‘ImageDataGenerator’
Step 10 Fit the model
Step 11 Evaluate the model
Step 12 Plot the loss and accuracy
Step 13 Print the Confusion Matrix
Step 14 Print the model summary


I have considered the dataset from Kaggle, referenced below, which is collection of textures of the histological images of human Colorectal Cancer.
• It contains 5,000 images divides as 4,000 images for training and 1,000 images for testing
• It has 8 classes of tissues namely, Tumor epithelium, Simple stroma, Complex stroma, Immune cells, Debris, Normal Mucosal glands, Adipose Tissue and Background (No Tissue)
• The images are of the size 150*150 pixels
• The images are all RGB i.e. they contain 3 channels, Red, Green and Blue
• It has 625 images of each class of tissue


The Convolutional Neural Network designed for Colorectal Cancer diagnosis from the histological images is successfully designed with an accuracy of 90.20% and a loss of 30.37%. The loss of this model is significantly high and suggests overfitting. This can be further improved by increasing the dropout or by reducing the size of the model. Also, Batch Normalization layer can be added to normalize the inputs of each of the layers.


“Colorectal Histology MNIST”. Kaggle. Available at: https://www.kaggle.com/kmader/colorectal-histology-mnist 
