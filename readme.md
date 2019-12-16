## CIS 5543 Computer Vision project (in-progress)  
Credits:  
We use SRGan code from this paper: https://arxiv.org/abs/1609.04802  
Original code for SRGan: https://github.com/tensorlayer/srgan  
VIRAT Dataset: https://data.kitware.com/#collection/56f56db28d777f753209ba9f  


## Usage  

The dataset we are using is approximately 70GB. To download, run:
```bash
bash download_dataset.sh
```
This will download the dataset and unzip it in the `data` folder. This script requires the wget package. To install: `sudo apt install wget`  

We containerized the environment. Experiments can be run using Singularity:  
Create a singularity image:
```bash
singularity pull docker://tensorflow/tensorflow:2.0.0-gpu-py3
singularity build --sandbox container/sandboxdir tensorflow_2.0.0-gpu-py3.sif
```
And shell into the image:
```bash
singularity run --writable container/sandboxdir
# install required python packages
pip3 install -r srgan/requirements.txt
```
Running SRGAN experiment:
```bash
exit # exit the singularity image
# run the job script
cd srgan
bash job_script.sh
```
This will create the folder srgan/outputs.  
In the folder there will be 4 folders with today's timestamp and experiment name.  
Each experiment folder will include outputs from every 10 iterations, one iteration being 96 samples processed from one video in ground video dataset.  
Evaluation:
```bash
# shell into the image again
singularity run --writable container/sandboxdir

# run code in evaluation mode
python3 train.py --inputsize 36 --exp input36 --mode evaluate
python3 train.py --inputsize 48 --exp input48 --mode evaluate
python3 train.py --inputsize 64 --exp input64 --mode evaluate
python3 train.py --inputsize 96 --exp input96 --mode evaluate
```
This will feed an image we picked from aerial dataset to each of the networks and save the generated image along with the input image in each of the experiment directories