# Master Thesis Project (Novel View Synthesis of Semi-Transparent Volumetric Datasets)

## Abstract

The task of view synthesis i.e. generating novel views of a scene or object using an existing set of views has gained a lot of attention in recent years due to its applications in virtual and augmented reality. Recent work by Mildenhall et al.~\parencite{mildenhall2020nerf} has shown impressive results on the task of view synthesis using \gls{mlp} to represent a 3D scene as a continuous function. Representing scenes as continuous functions instead of using discrete representations like voxel grids has a huge advantage in terms of memory usage. To render high-quality images, the voxel grids need to be sampled at high resolutions in which case the memory required to store the scene information increases exponentially. Using implicitly defined continuous functions parameterized by \glspl{mlp} does not suffer from this disadvantage. Mildenhall et al. in their work titled \gls{nerf}~\parencite{mildenhall2020nerf} generate a ray passing through a volume for each image pixel. Then they sample multiple points on each ray and use classical volume rendering techniques to calculate the final color of each ray. Though their approach shows impressive visual results, it is still expensive in the generation of points for each ray. To get the color value for one pixel, the network has to learn the color and opacity information on multiple points in the volume. We present an approach to represent an object on a bounding sphere around it instead of taking a whole volume to represent a 3D object. We project the images of the object on the bounding sphere. The intuition behind this approach is that how the image information changes across the bounding sphere give an understanding of how the object appearance changes in the 3D world. We provide an analysis of how this approach works on the task of view synthesis for semi-transparent objects. 

--------------------

The details of the experiments done in the thesis are explained in detail in the report.
We recommend to kindly go through the report first to understand the experiments.

Four different networks have been trained on three different datasets. We use 'hydra' library
to manage the configuration to be run. Example configuration files can be seen in config folder

## Structure of config folder:

> data -> One config file for each dataset to be used  
> models -> One config file for each model configuration  
> train -> All other configuration parameters can be specified in the config file in this directory  
> config.yaml -> Specify the combination of configuration files to be run  
> ### Model Configurations:
> ConvNet -> Model for training on RGB images or isosurface. Model predicts the RGB image.  
> Conv2Net -> Model for training on RGB images or isosurface. Model predicts the alpha image and RGB image.  
> NeRF -> Model for nerf approach trained only on 3D points.  
> NeRF2 -> Model for nerf approach trained on both 3D points and view directions  
> ### Example configuration files for training:
> run_no_map -> Trained on our raw input representation  
> run32.0_256 -> Trained on Fourier Encoded data with SD of 32.0 and feature size of 256  

## Running the code

    Specify the configuration combination to run in config.yaml and run main.py
