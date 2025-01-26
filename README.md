# CNN for Hand-Based Binary Gender Classification

This repository contains the implementation of a convolutional neural network (CNN) system designed to classify gender based on hand images. 
Leveraging the LeNet and AlexNet architectures, the system focuses on hand morphology and texture, achieving high classification accuracy through preprocessing and model optimization.
The project was developed as part of the "Fundamentals of Data Science" course during the Master's degree in Computer Science at Sapienza University of Rome in the 2024/25 academic year.

## 🖥️ Steps

1. Clone the repository

2. Install dependencies

3. Download the 11k Hands dataset
   - [Dataset Link](https://sites.google.com/view/11khands)

4. In the main file, set the necessary values for execution:
     - num_exp: Number of epochs
     - image_path: Path to the folder containing the images of the 11kHands dataset
     - cav_path: Path to the file containing the metadata of the 11kHands dataset images
     - num_train: Number of images to use for the training phase
     - num_test: Number of images to use for the testing phase

5. Run the main script


## 🧾📈 More details and Performance evaluation
For more details on the implementation and its performance, you can refer to the paper [HandsGenderRecognition](https://github.com/patriziorenelli/HandsGenderRecognition/blob/main/HandsGenderRecognition.pdf)

# 🧑‍💻 Collaborators

- Matteo Rocco
- Alessio Taruffi
- Nicolò Candita
