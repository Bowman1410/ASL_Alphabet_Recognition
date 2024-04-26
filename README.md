# ASL Alphabet Recognition
This repository contains code for building a convolutional neural network (CNN) model to recognize American Sign Language (ASL) alphabets using TensorFlow and Keras.

## Dataset
The ASL Alphabet dataset used in this project consists of images representing each letter of the ASL alphabet in American Sign Language. The dataset is organized into three sets: training, validation, and testing. The dataset is availble at: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Requirements
* opencv-python==4.7.0.72
* pillow==9.4.0
* numpy==1.24.2
* scikit-learn==1.2.1
* split-folders==0.5.1
* tensorflow==2.11.0
* matplotlib==3.7.0

## Usage
1) Clone the repository:
``` bash
git clone https://github.com/your-username/ASL_Alphabet_Recognition.git
cd asl-alphabet-recognition
```
2) Install dependencies:
``` bash
Copy code
pip install -r requirements.txt
```
3) Run the code:
```bash
Copy code
python asl_alphabet_recognition.py
```
## Results
The model achieves an accuracy of 0.969 on the test dataset.

## Author
Iain Bowman

## License
This project is licensed under the MIT License.
