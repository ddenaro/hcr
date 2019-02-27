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
![Confusion matrix](https://github.com/ddenaro/hcr/blob/master/raw/emnist_confusion.png)

### Step 2 - Deploy your neural network
Deploy a neural network to an MCU it's not a trivial task. During the NN design you have used a Deep Learning library running on your desktop or in the cloud, now it's time to move on the edge. In order to traslate your trained NN in an STM32 code we leveranging a tool made by ST the [STM32CubeMX with the Cube.AI extension](https://www.st.com/en/embedded-software/x-cube-ai.html).\
Install the STM32CubeMX and then:
1. Create a new Board project for the STM32L496G Discovery board
2. Initialize all peripherals with their defaults mode
3. Press the Additional softwares button and select the Cube.AI/Core pack, press OK and the package is added to the project
4. Expand Additional software from the configuration tree and select the Cube.AI
5. Enable Artificial Intelligence Core check box, this enable the tool
6. Now use the configuration panel to insert your NN
    - Name = hcr_nn
    - Model kind = Keras
    - Select saved model as model format
    - Use the browse button to insert the model filename
    - Then press the Analyze button
7. The model analysis shows you some useful information about:
    - Computation cost (Complexity) in [MACC](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)
    - Flash occupation for the network weights
    - RAM required to run the network
8. For the code generation use the Project Manager panel to select the SW4STM32 toolchain and press the GENERATE CODE button
9. In the destination directory the AI folder under Middlewares > ST contains your NN traslated in C code

### Step 3 - Application integration
The NN landing to the edge is almost done. It's time to integrate the AI into your application.
In this repository you can find the complete application for the handwritten character recognition.
You can use the [SW4STM32](https://www.st.com/en/development-tools/sw4stm32.html) free ide to compile and debug the project.\
The Cube.AI has traslated the NN into a static library driven by a couple of API. Using this interface is very simple, first of all you have to allocate some RAM for the input tensor ( your input buffer ), the output sensor ( the output buffer for instance the result of the softmax layer ) and the activation memory ( the memory used to run an inference ).\
Then you have to initialize the NN engine:
```C
    ai_hcr_nn_create(&g_hcr_network,(const ai_buffer*)AI_HCR_NN_DATA_CONFIG);
    ai_network_params net_params = AI_NETWORK_PARAMS_INIT(
                                   AI_HCR_NN_DATA_WEIGHTS(ai_hcr_nn_data_weights_get()),
                                   AI_HCR_NN_DATA_ACTIVATIONS(g_net_activations));
    ai_hcr_nn_init(g_hcr_network,&net_params);
 ```
 The code below initializes the network handle g_hcr_network with the NN weights and the activation memory.
 To run an inference simply call:
 ```C
 ai_hcr_nn_run(g_hcr_network, &g_net_in[0], &g_net_out[0]);
 ```
 Where g_net_in and g_net_out are the input and output tensors.








