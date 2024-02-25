# E2E-SPTTN
This is repository for E2E-SPTTN: Spatial–temporal transformer for end-to-end sign language recognition.
![image](https://github.com/Nice-Zhang66/E2E-SPTTN/assets/72641963/3e345af5-d8db-4040-9202-e0ecebd07e9c)

# Installation
1. Create a virtual environment conda create -n spttn python=3.10 -y and activate it conda activate spttn
2. Install Pytorch 2.0
3. git clone https://github.com/Nice-Zhang66/E2E-SPTTN
# Get Started 
Download the dataset: You can choose any one of following datasets to verify the effectiveness of TLP.

## PHOENIX2014 dataset
Download the RWTH-PHOENIX-Weather 2014 Dataset [https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/]. Our experiments based on phoenix-2014.v3.tar.gz.

After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014

The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.

cd ./preprocess
python data_preprocess.py --process-image --multiprocessing
## CSL dataset
The results of TLP on CSL dataset is placed in the supplementary material.

Request the CSL Dataset from this website [https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html]

After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
ln -s PATH_TO_DATASET ./dataset/CSL

The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.

cd ./preprocess
python data_preprocess-CSL.py --process-image --multiprocessing
