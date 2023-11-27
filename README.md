# ESCAPE: Energy-based Selective Adaptive Correction for Out-of-distribution 3D Human Pose Estimation


## Setup

### Environment
1. Create a new conda env from environment.yml: ```conda env create -f environment.yml```
2. Activate the environment: ```conda activate escape_env```

### Data
<!-- Note: for GDrive data use [this script](https://stackoverflow.com/questions/37453841/download-a-file-from-google-drive-using-wget) to download files.
1. 3DHP 
- Download the [HybrIK](https://github.com/Jeff-sjtu/HybrIK) version of the annotations from [GDrive](https://drive.google.com/drive/folders/1Ms3s7nZ5Nrux3spLxmMMAQWc5aAIecmv)
- use scripts/unpack_3dhp_frames.py to unpack the 3DHP videos into imgs -->


For training and evaluating the networks in ESCAPE, we require the backbone network predictions on HP3D, MPII, PW3D. 

Using our preprocessed data:
1. Download data ready to use with ESCAPE: [GDrive](https://drive.google.com/file/d/1_NXaw4WYcGvWPgmd9iTqe-4XC14zGCK_/view?usp=sharing)
2. Extract the data and modify config.cnet_dataset_path in config.py to point to the folder 

### Checkpoints

1. Download the trained CNet and RCNet checkpoints: [GDrive](https://drive.google.com/file/d/13vKRLuwvxXQqAyNccuAh-mYc8WTOvsx3/view?usp=sharing)
2. Extract and place at ./ckpts

## Usage

Modify options in config.py to select experiments, change backbones, and modify other parameters. 

Run experiments.py to execute the tasks in config.tasks: ``` python3 experiments.py ```

Important Options:
- **config.tasks:** is a list which controls what experiments will be performed
- **config.backbone:** changes the pre-trained backbone estimator being used
- **config.test_adapt:** enables intensive test-time adaptation, else only fast correction will be applied
- **config.TTT_e_thresh:** enables Energy-based sample selection, instead of adapting to all samples
- **config.energy_thresh:** changes the Energy threshold used for Energy-based sample selection




