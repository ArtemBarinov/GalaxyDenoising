# GalaxyDenoising


## Getting Started
------
### Requirements
* Python 3.6
* tensorflow-gpu==1.9.0
* Keras==2.2.0
* numpy==1.14.0+mkl
* h5py==2.8.0
* matplotlib==2.2.2

A full list of packages installed when this code was created is available in REQUIREMENTS.md

### Running the Code

The following instruction are taken from https://github.com/ChristopherAkroyd/Galaxy-Classification/ :

1. Download the raw Galaxy Zoo csv data from [The Galaxy Zoo website](https://data.galaxyzoo.org/), the specific data set we use is the one generated with the methods described by Hart et al. (2016) which can be downloaded from this [Direct Link](http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz).
2. Extract the Zip file and put the CSV within the data folder.
3. Run `python download_gal_zoo_cutouts.py` to download the images used from the Sloan Digital Sky Survey website. (This process can take upwards of 4 hours.)

4. The project file and notebook contain all the necessary files required to run the train the models as long as all the
python packages are successfully installed.

### Structure
##### Folder Layout
* data
    * elliptical - A collection of elliptical galaxy images.
        * test - a testing subset of the elliptical images
        * train - a training subset of the elliptical images
    * spirals - A collection of spiral galaxy images.
        * test - a testing subset of the spiral images
        * train - a training subset of the spiral images
    * galaxy_data - A whole dataset of images.
        * test - a testing subset of the dataset
        * train - a training subset of the dataset
* results - contains all the trained Models, with the following structure:
    * Elliptical/whole dataset
        * Greyscale/RGB
            * Model type
* src
    * models - Stores all the implemented models.
* Notebook - the jupyter notebook to run the code.

### Comments
The models are named ks for a block of two convolutional layers followed by dropout and batch normalisation.
ks signifies that kernel_size, signifying that kernel size can be changed. x2-x6 signifies the amount of the basic blocks the
encoder and decoder contain individually. Please note, that as the structure of the code for all the models is largely similar, only
the Baseline and Multi-Filter models are fully commented.
