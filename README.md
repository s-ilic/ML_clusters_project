# Machine-learning codes for galaxy clusters in cosmology

**Original author**: Stéphane Ilic

**Contributors**: Simona Mei

## Overview

This repository currently contains 5 directories, whose contents are as follows:

* `classic_CNN`: "classic" Tensorflow-based convolutional neural networks (original design & XXX), dedicated to detecting galaxy clusters and determining their redshift from raw (SDSS) images;
* `classic_NN`: "classic" Tensorflow-based sequential neural networks (original design), dedicated to determining the position and redshift of galaxy clusters from (SDSS) galaxy catalogues; 
* `invariant_NN`: permutation-invariant neural networks written in Tensorflow, inspired by the Janossy architecture of [Murphy et al. 2019](https://arxiv.org/abs/1811.01900), dedicated to determining the position, redshift and richness of galaxy clusters from (SDSS) galaxy catalogues; 
* `YOLOV3`: Tensorflow-based convolutional neural networks, following the YOLOv3 architecture of [Redmon et al. 2018](https://arxiv.org/abs/1804.02767), dedicated to the detection of galaxy clusters from raw (SDSS) images;
* `misc_tools`: collection of useful script, mostly for retrieving SDSS images and constructing bounding boxes to be given as input for the YOLOv3 networks.

## Prerequisites, installation, and usage

To be written.
test
