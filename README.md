# Optimizing-iFormer

In this experiment, we aim to enhance the iFormer training process by incorporating Data Parallelism, Distributed Data Parallelism, Model Parallelism, and DL Pipelining.


## Project Milestones

1. Setup GitHub Repository ✅
2. Configure dataset and environment in slurm ✅
3. Model training with single GPU and benchmarking ✅
4. Implement and execute DataParallel (DP) with 4 GPUs ✅
5. Implement and execute Distributed Data Parallel (DDP) with 4 GPUs ✅
5. Implement and execute model paraallelism (pipeline approach) with 4 GPUs ✅
7. Final slides and presentation 🚧


## Data setup

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


## Environment and Repo Setup

1. Clone repo
    ```
    git clone https://github.com/TianjianNi/Optimizing-iFormer.git
    ```

2. Edit the ```DATA_PATH``` variable in the .env file.

    For example, you can set your env file as below
    ```
    DATA_PATH=/scratch/NETID/project/cifar10-128
    ```

## Running Benchmark

- model code: model.py
- main code: main_DP.py (1 GPU allocation with batch file)

Benchmark.sh sample
```
#!/bin/bash
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=Benchmark
#SBATCH --output=Benchmark

module purge
singularity exec --nv \
            --overlay /scratch/tn2151/pytorch-example/overlay-10GB-400K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
        python main_DP.py;"

```

Slurm command
```
sbatch Benchmark.sh
```

## Running DataParallel (DP)

- model code: model.py
- main code: main_DP.py


DP.sh sample

```
#!/bin/bash
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=DP
#SBATCH --output=DP

module purge
singularity exec --nv \
            --overlay /scratch/tn2151/pytorch-example/overlay-10GB-400K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
        python main_DP.py --batch_size 512 --num_workers 16;"
```
Slurm command
```
sbatch DP.sh
```


## Running Distributed Data Parallel (DDP) 

- model code: model.py
- main code: main_DDP.py


DDP.sh sample

```
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=DDP
#SBATCH --output=DDP

module purge
singularity exec --nv \
            --overlay /scratch/tn2151/pytorch-example/overlay-10GB-400K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
        python main_DDP.py;"
```

Slurm command
```
sbatch DDP.sh
```

## Running Model Parallelism (Pipeline)

- model code: pipe_model.py
- main code: main_PIPE.py

PIPE.sh sample

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=PIPE
#SBATCH --output=%x.out

module purge

singularity exec --nv \
            --overlay /scratch/dy2242/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
            python main_PIPE.py
            "
```

Slurm command
```
sbatch PIPE.sh
```

## Results

### Data Parallelism

| Epoch 5 Result (Use first 4 epochs as warmup) | Benchmark on 1 GPU | DP on 4 GPUs | DDP on 4 GPUs |
|----------------------------------------------|--------------------|--------------|-------------|
| Dataloading Time (sec)                       | 6.201              | 7.050        | 8.861       |
| Training Time (sec)                          | 371.445            | 104.738      | 95.625      |
| Running Time (sec)                           | 378.163            | 112.506      | 105.318     |
| Speedup (T benchmark / T approach)           | -                  | 3.36         | 3.59        |


### Model Parallelism

| Metric                                     | Benchmark on 1 GPU | Model Parallelism with Pipeline |
|--------------------------------------------|--------------------|---------------------------------------------|
| Dataloading Time (sec)                     | 6.201              | 6.888                                       |
| Training Time (sec)                        | 371.445            | 211.158                                     |
| Running Time (sec)                         | 378.163            | 220.429                                     |
| Speedup (T benchmark / T approach)         | -                  | 1.72                                        |

### Observation

In assessing distributed approaches during Epoch 5, it becomes apparent that significant speedups can be attained across varied configurations. Notably, the use of Distributed Data Parallelism with 4 GPUs yields the most substantial speedup compared to alternative methods. Nevertheless, it is essential to recognize that employing a single GPU results in the highest accuracy, emphasizing the inherent trade-off between speedup and precision. This highlights the importance of thoughtfully weighing performance improvements together with accuracy concerns when selecting a distributed method. The best choice depends on the particular goals and priorities of the computational task.