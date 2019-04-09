Monocular depth estimation
============================

This repo is a simple Pytorch implementation of depth estimation that allows some flexibility in the model through command-line arguments.

The code skeleton is adapted from (https://github.com/fangchangma/sparse-to-dense), although the implementation is different from the paper.

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Testing](#testing)

## Requirements
This code was tested with Python 3.7, PyTorch 1.0.1, and CUDA 9.0.
- Install [PyTorch](http://pytorch.org/) on a machine with CUDA GPU.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and other dependencies (files in our pre-processed datasets are in HDF5 formats).
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py matplotlib imageio scikit-image opencv-python
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset in HDF5 formats, and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space, and KITTI requires 81G.
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
 	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
	cd ..
	```
	
- CARLA Dataset is an original dataset I made while driving in CARLA driving simulator (http://carla.org/). Some samples are available [here](https://drive.google.com/open?id=1y06fdYojxuADoHCBzkSUZXVKrx8MLlBm).
## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

For instance, run the following command to train a network with ResNet50 as the encoder, deconvolutions of kernel size 3 as the decoder and l1 loss for NYU V2 dataset. These three arguments can of course be varied.
```bash
python3 main.py -a resnet50 -d deconv3 -c l1 --data nyudepthv2
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```





