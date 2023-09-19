# NN-CA7-1.Cryptocurrency-Price-Estimation-Using-LSTM-2.Violence-Detection-in-Videos-Using-ResNet50

### 1.Cryptocurrency Price Estimation Using LSTM [Link](#part-1-cryptocurrency-price-estimation-using-lstm)

### 2.Violence Detection in Videos Using ResNet50 [Link](#part-2-violence-detection-in-videos-using-resnet50)

# Part 1: Cryptocurrency Price Estimation Using LSTM

## Overview

This repository contains code and data for a study on estimating cryptocurrency prices using LSTM and GRU models. The models are implemented and evaluated on Litecoin and Monero datasets up to 2023.

## Models

### LSTM vs. GRU

- LSTM is a more complex architecture compared to GRU. It consists of a memory cell and three gates (reset gate, update gate) whereas GRU has three gates (input gate, forget gate, output gate).

- LSTM uses a memory cell to store information for long-term dependencies. The forget gate determines what information in the memory cell should be retained, and the input gate determines what new information to store.

- GRU also has a memory element, but it lacks a separate memory cell. Instead, it uses an update gate to decide how much of the past information to forget and how much of the new information to store.

- GRU is computationally less expensive than LSTM due to its simpler architecture. However, LSTM requires more time and data for convergence.

- In practice, the performance of LSTM and GRU depends on the specific task and dataset.

### Model Usage

In this study, both LSTM and GRU architectures are used. Two LSTM layers and one GRU layer process input data in parallel. A linear layer is then used to calculate the predicted price.

## Dataset and Preprocessing

- Datasets for both cryptocurrencies are obtained from their respective sources, covering data up to 2023. The 'Price' is selected as the target feature.

- Preprocessing includes max-min normalization of the data.

## Model Training

- Both LSTM and GRU models are implemented and trained for 100 epochs.

- No validation dataset is used, and Early Stopping is not applied.

## Results

### 1-Day Window

| Model    | Currency | MSE    | RMSE  | MAE  | MAPE  |
| -------- | -------- | ------ | ----- | ---- | ----- |
| LSTM     | Litecoin | 82.33  | 9.07  | 5.37 | 0.06  |
|          | Monero   | 162.31 | 12.74 | 7.61 | 0.06  |
| Proposed | Litecoin | 72.62  | 8.52  | 5.04 | 0.06  |
|          | Monero   | 123.30 | 11.10 | 6.76 | 0.006 |

### 3-Day Window

| Model    | Currency | MSE    | RMSE  | MAE  | MAPE |
| -------- | -------- | ------ | ----- | ---- | ---- |
| LSTM     | Litecoin | 144.43 | 12.02 | 6.43 | 0.07 |
|          | Monero   | 237.28 | 15.40 | 8.87 | 0.06 |
| Proposed | Litecoin | 146.86 | 12.12 | 6.37 | 0.07 |
|          | Monero   | 214.04 | 14.63 | 8.55 | 0.07 |

### 7-Day Window

| Model    | Currency | MSE    | RMSE  | MAE   | MAPE |
| -------- | -------- | ------ | ----- | ----- | ---- |
| LSTM     | Litecoin | 256.08 | 16.00 | 8.68  | 0.09 |
|          | Monero   | 413.52 | 20.34 | 11.51 | 0.08 |
| Proposed | Litecoin | 265.06 | 16.28 | 8.71  | 0.09 |
|          | Monero   | 393.44 | 19.84 | 11.41 | 0.09 |

## Conclusion

Both LSTM and GRU models are evaluated for estimating cryptocurrency prices. Results suggest that the performance of these models can vary based on the specific task and dataset. LSTM tends to perform better in tasks where long-term memory is crucial.

For more detailed information, please refer to the article.

# Part 2: Violence Detection in Videos Using ResNet50

## Data Acquisition and Preprocessing

In this project, in the DataLoader class, for each batch, we perform the following tasks:

1. Determine the label for the film.
2. Load the film and extract its frames.
3. Then, for the desired number of frames (i.e., 10), we perform the following tasks:

   - Extract two consecutive frames, subtract them, and save the result. The reason for this step is that, in the meantime, we also save 10 frames from both classes (violence and non-violence) from a film in one of the pre-defined folders, resulting in the following structure.

   Finally, we skip frames in intervals.

## Preprocessing Steps as Described in the Article

For the preprocessing steps mentioned in the article, we do the following:

1. Crop the image on both sides where it is black.
2. Change the size of the image to (256, 256).
3. Flip the image randomly both horizontally and vertically with a 50% probability.
4. Finally, normalize the image with a mean of zero and variance of one.

## Model Implementation and Training

The input size is (3, 256, 256). The model implementation involves using pre-trained weights from a 50-layer ResNet. After that, we use a D2convLSTM layer with 256 filters and a kernel size of 3, which enables the model to learn dependencies between changes and previous actions. After batch normalization and flattening, we use fully connected layers with 1000, 256, and 10 neurons respectively, using ReLU, ReLU, ReLU, and sigmoid activation functions for classification into the two classes: violence and non-violence. As per the article, we use the RMSprop optimizer with an initial learning rate parameter of 0.0001. After experimentation, it was observed that this learning rate provided smoother learning compared to 0.001, which was also mentioned at the end of the article. Due to the classification nature of the problem, binary cross-entropy is used.

## Data Splitting for Training, Validation, and Evaluation

The data is divided into three sets for training, validation, and evaluation, with respective usage percentages of 80%, 5%, and 15%. During the training process, EarlyStopping with a parameter of 5 is used, which leads to the training process stopping at the 25th epoch. Additionally, a learning rate schedule is used to dynamically adjust the learning process, starting from 0.0001 and decreasing to 0.00000325.

## Results

Based on the above results, we achieved good accuracy for both training and validation data. Additionally, for other metrics, we achieved 68% precision and recall on the validation data, resulting in a 68% score_1f for the validation data as well. The lack of smoothness for the validation data is due to the lower number of samples in this category (25 videos). In contrast, in the training data where the count is higher (400 videos), the various assessment measures show a smoother increase, leading to a reduction in loss.

Finally, the results for the test data are as follows:

- Evaluation Metrics for Test Data

In the evaluation data, which consists of 75 videos, we achieved an accuracy of 94%, and other measured values such as precision, recall, and score_1f are 73%, 74%, and 73% respectively. This demonstrates a good accuracy relative to the number of available data.

Moreover, the confusion matrix for all three data categories (training, validation, and evaluation) clearly shows how each video was categorized.
