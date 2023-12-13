# Optimizing-iFormer

In this experiment, we aim to enhance the iFormer training process by incorporating Data Parallelism, Distributed Data Parallelism, Model Parallelism, and DL Pipelining.

## 1. Data setup

We have selected CIFAR-10 as our training dataset, a widely used collection for image classification. However, the original CIFAR-10 images have dimensions of 32x32, which are inadequate for resizing to 224x224 for the iFormer Small model. To address this limitation, we will employ 128x128 Resized CIFAR-10 images obtained through CAI Super Resolution.

To download the dataset, you can also follow the steps outlined in 
https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/


1. Navigate to the Accounts page on Kaggle: https://www.kaggle.com/<USER_NAME>/account and create a new API token. This will create API credentials in json 

    ```{"username":<USER_NAME>,"key":<API_KEY>}```

2. On Greene CLI, run the following commands:

    ```
    pip3 install --user kaggle
    mkdir ~/.kaggle
    cd ~/.kaggle
    ```

3. Create the Kaggle API credential file (kaggle.json) using a text editor like Vim:

    ```
    vim kaggle.json
    {"username":<USER_NAME>,"key":<API_KEY>}
    :wq!
    ```

4. Go to your project folder and run
    ```
    kaggle datasets download -d joaopauloschuler/cifar10-128x128-resized-via-cai-super-resolution
    ```

5. Unzip the downloaded file
    ```
    unzip cifar10-128x128-resized-via-cai-super-resolution.zip
    ```


## 2. Environment Setup

Edit the ```DATA_PATH``` variable in the .env file.

## 3. temp

## 4. temp

## 5. Results

Benchmark: Epoch 5/5 Data loading time (sec) is 68.025 Training time (sec) is 367.235 Running time (sec) is 435.714 Accuracy is 75.972%

DP 4GPU: Epoch 5/5 Data loading time (sec) is 43.091 Training time (sec) is 130.066 Running time (sec) is 174.626 Accuracy is 76.298%

DDP 4GPU: Epoch 5/5 Data loading time (sec) is 15.763 Training time (sec) is 105.363 Running time (sec) is 123.452 Accuracy is 72.048%