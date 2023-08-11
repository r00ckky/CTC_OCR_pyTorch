## OCR using PyTorch

End to end variable-length English character dectection and recogintion based on pyTorch 

#### Model
* The model utilizes the CRNN (Convolutional Recurrent Neural Network) architecture, along with CTC loss function. 

#### Input
* The OCR takes in an image of size (64, 256) and then using CNN extracts features from it which are then feed into the RNN to classify the text.

#### Dataset
* The dataset has been taken form kaggle. [link](https://www.kaggle.com/datasets/landlord/handwriting-recognition)

#### Example Input

<a href="https://imgbb.com/"><img src="https://i.ibb.co/J7nsM6x/TRAIN-00012.jpg" alt="TRAIN-00012" border="0"></a>
