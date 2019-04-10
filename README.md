Plug-and-play architecture for monocular depth estimation
============================

This repo is a Pytorch implementation of depth estimation that allows plug-and-play style flexibility in the model (in particular, encoder, decoder and loss function) through changing command-line arguments. It is also straightforward to add more custom options.

The code skeleton is adapted from (https://github.com/fangchangma/sparse-to-dense), although the implementation is different from the paper as we are not introducing any sparse samples.

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Results](#results)
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
	cd .. #data directory should be in the same level as monodepth_pytorch directory!
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
 	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
	cd ..
	```
	
- CARLA Dataset is an original dataset I gathered while driving in CARLA driving simulator (http://carla.org/). There are ~10k photos recorded across 5 maps. The splitted (80:20) data is available [here](https://drive.google.com/open?id=145_-KAQVKYeKWtoL2Z8pmMypdVD7xDdn) (7.2GB).
## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

For instance, run the following command to train a network with ResNet50 as the encoder, deconvolutions of kernel size 3 as the decoder and l1 loss for NYU V2 dataset. These three arguments can of course be varied.
```bash
python3 main.py -a resnet50 -d deconv3 -c l1 --data nyudepthv2
```

In summary, this code allows
```bash
python3 main.py -a {resnet18 or resnet50} -d {deconv2 or deconv3} -c {l1 or l2} --data {nyudepthv2 or kitti or carla}
```
It is of course possible to add new options to this.

You can also vary hyperparameters such as number of epochs, batch size, learning rate, momentum and weight decay through command-line arguments. See `--help` for the full set of options.



## Results
Results are automatically tracked every epoch in both Tensorboard and in a csv file. Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```
To run tensorboard, run
```bash
tensorboard --logdir=tensorboardresults/[model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```
