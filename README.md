# Colorectal-Cancer-Detection
Colorectal Cancer Detection Using Deep Learning

## Algorithm
1. Import NumPy, TensorFlow and other required libraries
1. Read the data from a csv file
1. Check the shape and datatype
1. Reshape
1. Split the data into training and testing dataset
1. Using to_categorical in keras to get a binary class matrix from vector class
1. Define the model architecture
1. Compile the model using optimizer
1. Increasing the dataset using ‘ImageDataGenerator’
1. Fit the model
1. Evaluate the model
1. Plot the loss and accuracy
1. Print the Confusion Matrix
1. Print the model summary


I have considered the dataset from Kaggle, referenced below, which is collection of textures of the histological images of human Colorectal Cancer.
• It contains 5,000 images divides as 4,000 images for training and 1,000 images for testing
• It has 8 classes of tissues namely, Tumor epithelium, Simple stroma, Complex stroma, Immune cells, Debris, Normal Mucosal glands, Adipose Tissue and Background (No Tissue)
• The images are of the size 150*150 pixels
• The images are all RGB i.e. they contain 3 channels, Red, Green and Blue
• It has 625 images of each class of tissue


The Convolutional Neural Network designed for Colorectal Cancer diagnosis from the histological images is successfully designed with an accuracy of 90.20% and a loss of 30.37%. The loss of this model is significantly high and suggests overfitting. This can be further improved by increasing the dropout or by reducing the size of the model. Also, Batch Normalization layer can be added to normalize the inputs of each of the layers.


“Colorectal Histology MNIST”. Kaggle. Available at: https://www.kaggle.com/kmader/colorectal-histology-mnist 
