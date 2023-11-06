### ESCAPE: Energy-based Selective Adaptive Correction for Out-of-distribution 3D Human Pose Estimation

Cool things, I swear.


## Data
Note: for GDrive data use [this script](https://stackoverflow.com/questions/37453841/download-a-file-from-google-drive-using-wget) to download files.
<!-- 1. 3DHP
- Download the data from [SPIN](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) (which have been [preprocessed](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)) using:
``` 
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
```
- We only care about "mpi_inf_3dhp_train.npz" and "mpi_inf_3dhp_valid.npz" -->
1. 3DHP 
- Download the [HybrIK](https://github.com/Jeff-sjtu/HybrIK) version of the annotations from [GDrive](https://drive.google.com/drive/folders/1Ms3s7nZ5Nrux3spLxmMMAQWc5aAIecmv)
- use scripts/unpack_3dhp_frames.py to unpack the 3DHP videos into imgs
