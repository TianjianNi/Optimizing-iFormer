# Optimizing-iFormer
HPML Project Here we go

On your Greene, 

cd /scratch/NetID/
mkdir project

Download dataset from https://www.kaggle.com/datasets/joaopauloschuler/cifar10-128x128-resized-via-cai-super-resolution/data

scp CIFAR\ 10.zip NetID@dtn.hpc.nyu.edu:/scratch/NetID/project

unzip CIFAR\ 10.zip 

cd cifar10-128/

Modify downloaded data file

cifar10-128
|-- train
|   |-- class0
|   |   |-- imgxxx.png
|   |   |-- imgxxx.png
|   |   |-- ...
|   |-- class1
|   |   |-- imgxxx.png
|   |   |-- imgxxx.png
|   |   |-- ...
|   |-- ...
|   |-- class9
|       |-- imgxxx.png
|       |-- imgxxx.png
|       |-- ...
|-- test
    |-- (similar structure as train)

Results

Benchmark:
Epoch 5/5 
Data loading time (sec) is 68.025
Training time (sec) is 367.235
Running time (sec) is 435.714
Accuracy is 75.972%

DP 4GPU:
Epoch 5/5 
Data loading time (sec) is 43.091
Training time (sec) is 130.066
Running time (sec) is 174.626
Accuracy is 76.298%

DDP 4GPU:
Epoch 5/5 
Data loading time (sec) is 15.763
Training time (sec) is 105.363
Running time (sec) is 123.452
Accuracy is 72.048%

