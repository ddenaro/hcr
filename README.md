# AI to the edge
## The handwriting recognition tutorial 
This project is an hand on guide to deploy an artificial neural network to a low power MCU, in particular the target is to recognize digits and letters written using the touch screen of the [STM32L496G Discovery board](https://www.st.com/en/evaluation-tools/32l496gdiscovery.html).

![Character recognition](https://github.com/ddenaro/hcr/blob/master/raw/i-0001.gif)

### Step 1 - Design your own neural network in Keras
Recognizing handwritten digits is a well known problem for anyone beginning to study the artificial neural networks.\
The dataset used for training our network is the [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) and the model is a simple convolutional neural network (CNN).\
This [notebook](https://colab.research.google.com/drive/16YtnpdiDW0F3mPOXmZigrvgZRMvL9wpf) hosted on Google's Colaboratory allows you to train and evaluate the NN model, please feel free to modify the code in order to do your experiments.\
The notebook is divided in 4 parts:
1. Download EMNIST dataset - Download and unpack the EMNIST dataset
2. Database setup - Setup the dataset selecting digits and capital letters only
3. CNN Training - Setup the model and start the training ( remember to change the runtime type to GPU accellerated )
4. Evaluate and generate a confusion matrix - Evaluate the accuracy of the trained network and generate a confusion matrix


