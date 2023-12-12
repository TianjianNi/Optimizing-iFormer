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

custom_dataset
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
    |-- (similar structure as training)


Progresses:

