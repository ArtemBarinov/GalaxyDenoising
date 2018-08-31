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

The project file and notebook contain all the necessary files required to run the code as long as all the
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
