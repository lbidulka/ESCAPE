### Adaptable Inference for 3D Human Pose Estimation

Cool things, I swear.


## Data

1. 3DHP
- Download the data from [SPIN](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) (which have been [preprocessed](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)) using:
``` 
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
```
- We only care about "mpi_inf_3dhp_train.npz" and "mpi_inf_3dhp_valid.npz"