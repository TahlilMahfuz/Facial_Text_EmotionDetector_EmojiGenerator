## Emoji Generation Through Facial Emotion and Text Sentiment Analysis

## Table of Contents
- [Authors](#Authors)
- [Overview](#Overview)
- [Google Colab Link For Text Sentiment Detection](#Google_Colab_Link_For_Text_Sentiment_Detection)
- [Facial_Emotion_Detection_Dataset_Link](#Facial_Emotion_Detection_Dataset_Link)
- [Text_Emotion_Detector_Dataset_Link](#Text_Emotion_Detector_Dataset_Link)
- [Train_Facial_Emoiton_Detection_Model](#Train_Facial_Emoiton_Detection_Model)
- [Train_Text_Emotion_Detection_Model](#Train_Text_Emotion_Detector_Model)
- [Emoji_Generation](#Emoji_Generation)
- [Installation](#installation)
- [Contribution](#contribution)


## Authors

    Author_1 : K.M. Tahlil Mahfuz Faruk (SID: 200042158)
    Author_2 : Dayan Ahmed Khan (SID: 200042105)
    Author_3 : Shadman Sakib Shoumik (SID: 200042144)

## Overview

This project combines facial emotion and text sentiment analysis to generate emojis that match user emotions. The system uses a Convolutional Neural Network (CNN) for facial emotion detection and a Bidirectional Long Short-Term Memory (BiLSTM) model for text emotion detection. The goal is to enhance user experience and communication in digital interactions by accurately reflecting emotions through emojis.

## Google_Colab_Link_For_Text_Sentiment_Detection

    https://colab.research.google.com/drive/1ryjZXFMbpTz-Z6EXAySyBiy5Jd-oFHnL?usp=sharing

## Facial_Emotion_Detection_Dataset

    https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/
 
## Text_Emotion_Detector_Dataset

    Uploaded in the repository. 
        Folder Name: TextEmotionDetectionDataset 
        File Name: data_train.csv & data_test.csv 

## Train_Facial_Emoiton_Detection_Model

1. Download the dataset from the link provided above and extract the zip file.

2. Run the following command to train the model:

    python -u "\path\to\facial_emotion_detection.py"

3. A h5 extension file would be generated after the training is complete.

4. Use the h5 file to predict the emotion of a facial image in realtime_detection.py file.

5. Run the following command to predict the emotion of a facial image:

    python -u "\path\to\realtime_detection.py"

## Train_Text_Emotion_Detector_Model

1. Download the datasets data_test.csv & data_train.csv

2. Mount them in Google Colab
    (https://colab.research.google.com/drive/1ryjZXFMbpTz-Z6EXAySyBiy5Jd-oFHnL?usp=sharing)

3. Run the Google Colab Notebook Codes

## Emoji_Generation

    1. Run the following command to generate emojis:

        python -u "\path\to\Emoji_Generator.py"

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TahlilMahfuz/Facial_Text_EmotionDetector_EmojiGenerator.git

2. Install Depecdencies:
    
    pip install -r packages.txt

3. Run the applocation:

    python -u "\path\to\Emoji_Generator.py"

## Contribution

    #Facial_Emotion_Detection : K.M. Tahlil Mahfuz Faruk (SID: 200042158)

    #Text_Sentiment_Detection : Dayan Ahmed Khan (SID: 200042105)

    #Emoji_Generation : Shadman Sakib Shoumik (SID: 200042144)


    
    
    
