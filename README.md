# Machine-learning codes for galaxy cluster detection in cosmology

This repository is a sandbox of prototype machine-learning models and experiments for galaxy cluster detection and characterization in cosmology. It brings together several approaches -- including convolutional nets, object-detection networks, and catalogue-based networks -- to identify galaxy clusters (and their properties like redshift) in survey data. The focus is on exploring methods (rather than delivering a polished pipeline), using SDSS imaging and catalog data as inputs. In summary, the code covers end-to-end experiments from data preparation to training models for cluster detection in images and cluster-property inference from galaxy catalogues.

## Repository Structure

The code is organized into key directories, each containing related models or utilities:

- **classic_CNN**: Classical TensorFlow/Keras convolutional neural networks for **image-based cluster detection** (and planned redshift estimation) from raw SDSS images. These scripts implement binary classifiers to distinguish cluster vs. non-cluster images.
- **classic_NN**: "Classic" sequential (fully-connected) neural networks for **catalog-based inference**. These models take galaxy catalogue features (e.g. positions, magnitudes) as input and attempt to predict the location and redshift of clusters from those tabular data.
- **invariant_NN**: Permutation-invariant (Janossy-style) neural networks for **cluster property regression**. Inspired by Murphy et al. (2019), these networks aggregate unordered galaxy-member sets (from cluster catalogs) to predict cluster position, redshift, and richness in a way that does not depend on object ordering.
- **YOLOV3**: TensorFlow implementation of **YOLOv3 object detection** for finding clusters directly in images. This contains scripts (adapted from a YOLOv3 example) to train and run a one-shot detector that outputs bounding boxes around cluster regions.
- **misc_tools**: Utility scripts for **data preparation**. For example, there are Python scripts to query/download SDSS cutouts and to compute bounding boxes from cluster member lists (so as to generate label files for the YOLO pipeline).

Each directory contains the code and (in some cases) documentation specific to that approach. For instance, `classic_CNN` has image data generators and training scripts, while `invariant_NN` includes data-processing code and model implementations. The folders may also contain configuration files or sample notebooks (e.g. small Python scripts demonstrating how the data flows through the model).

## Installation

This project requires a modern Python 3 environment (tested on Python 3.7/3.8). The main dependencies include TensorFlow 2.x and common scientific libraries. You can install the necessary packages via `pip` or `conda`. For example:

```bash
# Using pip (TensorFlow 2.x, etc.)
pip install tensorflow numpy scipy matplotlib astropy healpy corner tqdm emcee
```

Alternatively, create a conda environment:

```bash
conda create -n ml_clusters python=3.8
conda activate ml_clusters
conda install numpy scipy matplotlib astropy healpy tqdm pip
pip install tensorflow corner emcee
```

**Notes**:
- **GPU support** is recommended for training the CNNs and YOLO models. Ensure CUDA/cuDNN are set up if using TensorFlow GPU.
- Some scripts may require extra tools (e.g. Image libraries like Pillow, or `wget` for downloading YOLO weights).
- No formal installer is provided; adjust import paths or data paths in the code as needed. (The original scripts assume certain directory structures like `SDSS_image_data/` or `SDSS_fits_data/`; you should set those paths to where your data are stored.)

## Usage

The repository contains scripts and (in some cases) notebooks to run experiments. Below are examples of how to run the core pipelines:

- **Classic CNN (image classifier)**: Organize your SDSS images into training/validation folders by class (e.g. `train/has_cluster/`, `train/no_cluster/`, and similarly for `valid/`). Then run the training script. For example:
  ```bash
  python classic_CNN/cnn_for_detection_multiGPU.py
  ```
  This will train a binary CNN on the images (resizing them as specified in the script, e.g. 1024×1024 with 3 channels) and periodically evaluate on the validation set. The script will output model weights, training logs, and can be adapted for multi-GPU via TensorFlow’s `MirroredStrategy`. After training, you can use the same script or similar code to make predictions on new images.

- **YOLOv3 Object Detector**: In the `YOLOV3` folder, follow the included instructions to train and test a YOLOv3 model. Typically, you would prepare two text files (`train.txt` and `valid.txt`) listing image paths and bounding boxes (the `misc_tools` can generate these from SDSS data). Then you can train the network:
  ```bash
  cd YOLOV3
  python train.py
  ```
  After training, run inference on validation images:
  ```bash
  python test.py
  ```
  A sample training/evaluation workflow (from the YOLO README) is:
  ```
  $ python train.py
  $ tensorboard --logdir ./data/log   # to monitor training
  $ python test.py
  $ cd mAP
  $ python main.py        # compute detection metrics (images saved in data/detection)
  ```
  These steps produce detected bounding boxes for clusters in test images and evaluate precision/recall. The resulting annotated images and mAP statistics help gauge detector performance.

- **Invariant NN (Janossy networks)**: The scripts in `invariant_NN` demonstrate building datasets from SDSS catalogues and training permutation-invariant models. For example, `nn_by_emcee.py` shows how galaxy member positions are permuted and fed through a small network (via emcee sampling). To train a model, you might use other scripts in that folder (e.g. `invariant_nn_v1.py`, not shown above) which define and train TensorFlow models. There are also plotting scripts (`plot_nn.py`, `plot_infer.py`) that visualize results of various model configurations.

- **Other Utilities**: The `misc_tools` folder includes scripts like `make_bb.py` (which computes bounding boxes from the redMaPPer FITS catalogs) and `make_empty_list.py` (which selects empty field images using a Healpix mask). Run these to prepare training data for YOLO. For example, running `make_bb.py` will read the SDSS redMaPPer catalogs (using Astropy FITS) and output lists of cluster image boxes. (The example paths in the script assume data is under `/home/users/ilic/ML/SDSS_fits_data/`.)

In general, expect to edit file paths and parameters in the scripts to match your setup. The output of these tools will be models (saved weights or logs) and result files (e.g. predicted labels, bounding box text files, plots of performance).

## Models and Methods

The project explores several machine learning approaches:

- **Convolutional Neural Networks (CNNs)**: Deep CNN classifiers built with Keras/TensorFlow, trained on galaxy image cutouts. These typically have a few convolutional+pooling layers followed by dense layers, ending in a sigmoid/softmax for cluster vs. non-cluster classification. They learn spatial features from the raw images.

- **YOLOv3 Object Detection**: A state-of-the-art single-shot detector adapted for clusters. YOLOv3 scans an image once and predicts bounding boxes and confidence scores for clusters. This pipeline uses publicly available YOLOv3 code (TensorFlow 2.x) to locate clusters in images. The model is trained end-to-end on images with annotated cluster boxes (generated from the redMaPPer member catalog).

- **Classic MLP (Fully-Connected) Networks**: Simple dense neural networks that take features from galaxy catalogues as input. For example, one could summarize a cluster’s galaxy list into a fixed-size feature vector (or use padding) and train an MLP to predict the cluster’s redshift or existence. These are implemented as standard TensorFlow Sequential models.

- **Permutation-Invariant (Janossy) Networks**: Advanced architectures that process sets of galaxies in a cluster in a way that does not depend on order. The code includes custom Janossy models (e.g. average over permutations of a subset of galaxies) to regress cluster properties like center coordinates, redshift, and richness. These networks often use a small "f" network applied to tuples of inputs, followed by an aggregation ("ρ") layer that combines permutation outputs. Various hyperparameters (number of galaxies, layers, activation, etc.) were experimented with, as seen in the `plot_nn.py` configurations.

- **Detection Pipelines**: Beyond the models themselves, the repository includes logic for end-to-end pipelines: data generators, preprocessing (e.g. normalizing images), loss functions (binary cross-entropy for classification, MSE for regression), and evaluation routines (e.g. computing true positive/false positive rates in `yolo_stats.py`).

Each approach is a **prototype**: code often contains example parameter settings (e.g. learning rates, image sizes) that may need adjustment, and many values are hard-coded for the original setup. Users should treat these as starting templates.

## Datasets

The experiments use astronomical survey data, primarily from the Sloan Digital Sky Survey (SDSS):

- **SDSS Image Cutouts**: Galaxy cluster image patches (e.g. 2048×2048 pixels at 0.396″/px) around known clusters. The scripts assume JPEG or NumPy array image files arranged in folders. The `misc_tools` include examples for downloading these images via SDSS APIs. (For YOLO training, cluster images are listed in `train.txt`/`valid.txt` along with bounding boxes.)
- **redMaPPer Catalogues (DR8)**: The redMaPPer cluster catalog and member galaxy catalog from SDSS DR8 are used as ground truth. These are standard FITS tables containing cluster IDs, redshifts, positions, and individual galaxy memberships. The code uses Astropy to read `redmapper_dr8_public_v6.3_catalog.fits` and the corresponding members file. From these, we obtain true cluster centers and define bounding boxes for training YOLO.
- **Empty Field Images**: A large set of SDSS "blank" fields (no known cluster) is generated via a Healpix mask to serve as negative examples for object detection. For example, `make_empty_list.py` scans SDSS tiles and selects those outside cluster regions using a Healpix footprint. These empty fields are included in YOLO validation to measure false positive rates.
- **Simulated Data**: (Optional) While most code is geared to real SDSS data, one can also feed in simulated sky maps or mock cluster images. For instance, experimenters might generate fake clusters or use cosmological simulations as input, as long as they can be formatted like the real data (FITS tables and image files).

In all cases, the user must prepare the data in the formats expected by the scripts (e.g. directory hierarchies for images, TXT files for bounding boxes). The code comments and utility scripts give guidance on the required formats (e.g. `x1,y1,x2,y2` coordinates in the YOLO label files).

## Development Notes

**Caveat**: This repository is a **work-in-progress** collection of exploratory prototypes. Many parts are not fully generalized or documented. For example, the top-level README itself notes "To be written" under installation and usage, indicating incomplete documentation. Users should be prepared to dig into the code, adjust hard-coded paths, and debug minor issues. The scripts were often developed for a specific user environment (e.g. file paths like `/home/users/ilic/...`); you will need to adapt these to your setup.

Because this is an *experimental* sandbox, expect that:
- Some hyperparameters (image sizes, learning rates, number of layers) were chosen ad hoc.
- Error handling and edge cases may be minimal.
- Not all features (e.g. multi-class classification or redshift regression) might be fully implemented even if mentioned.

Treat this project as a starting point: feel free to refactor, modularize, or extend it for your own research. Contributors are welcome to clean up scripts and add missing documentation.

## License

No open-source license file is provided with this repository. In other words, **no license** is explicitly specified. Users should be cautious about assuming any reuse rights; by default, reuse of code is subject to copyright and requires permission from the authors. (If you plan to publish or distribute derived work, contact the original author for clarification on licensing.)

