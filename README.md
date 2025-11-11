### Run Piscis in a Pipeline
The python script `spot_pipeline.py` wraps the [piscis](https://github.com/zjniu/Piscis) spot-calling algorithm to allow for high-throughput analysis of smFISH microscopy images in `nd2` or `tiff` format. \
Spots can be called on multi- or single-channel images as well as stacks of images (Z stack) or flat images. 

## Setup
#### File naming
The pipeline expects both image and mask files to contain the prefix `Location_XX_`, i.e. `Location_01_`, `Location_02_`, etc. The numbers need not be increasing or in numerical order, but this prefix **_must_** match between image files and their corresponding mask files. \
Example:
```
Image                        Mask
-----                        ----   
Location_01_A647_zStack.tif  Location_01_DAPI_MaxProj_cp_masks.tif 
Location_02_A647_zStack.tif  Location_02_DAPI_MaxProj_cp_masks.tif
Location_03_A647_zStack.tif  Location_03_DAPI_MaxProj_cp_masks.tif
...
```
Clone this repository to your local machine:
```
git clone https://github.com/Nuwah12/FISH-Spot-Calling.git
```
#### Dependencies 
Before running the pipeline, I suggest installing and activating the conda environment defined in the .yml file `imagingEnv.yml`. It contains all packages in correct versions needed to run it. \
To install, copy the `yml` file to your local machine and run: 
```
conda env create -f imgagingEnv.yml
conda activate bigfish
```
## Run Example _(Internal Use Only)_
To verify that everything is installed and working correctly, the `settings.yml` file is set up to work with image and mask files on `simurgh` at `/mnt/data0/noah/analysis/FISH-Spot-Calling/test_imgs` and `/mnt/data0/noah/analysis/FISH/Granta519/mask` respectively. To test, run
```
python3 piscis_pipeline.py settings.yml
```
The workflow will start with messages like
```
INFO:2025-11-11 10:58:50,803:jax._src.xla_bridge:822: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-11-11 10:58:50 INFO     Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
```
These are warnings and can be ignored. If there is no `jax` warning claiming that a GPU cannot be found, the pipeline is GPU accelerated. Confirm with `nvidia-smi` for NVIDIA GPUs.

## Usage
To execute the pipeline, run
```
python3 piscis_pipeline.py [-h] settings.yml
```
The script takes one argument, the `settings.yml` file. All parameters related to spot calling and image processing are in this file - no other command line arguments are needed. \
The available arguments include:
##### Image arguments
* `image_dir`: Path to directory with image files. Please ensure the files to be analyzed are in one directory.
* `image_type`: Type of image to be analyzed. Currently supported options are `tif`/`tiff` and `nd2`
* `channels`: An array with one entry for each fluoresence channel in the image. The dimension containing channel information is inferred based on the size of this list.
  * **NOTE:** Please ensure the ordering of channels here is the same as it is in your image file, i.e. in the below example, channel 0 is "DAPI", channel 1 is "Cy3", and channel 2 is "A647".
* `mask_dir`: Path to directory with mask image files in the same format as the image files and `image_type`. Refer above for how images and masks are mapped together.
##### Spot calling 
* `piscis_thresh`: The `threshold` parameter for the piscis model. It should not be changed much. Check [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC10862914/) for more information.
* `piscis_scale`: Piscis `scale` parameter
* `piscis_min_distance`: Piscis `minimum_distance` parameter
* `custom_model`: true/false specifying whether a custom model (file) is to be used
* `model_dir`: The directory (not the file) containing the custom modle file.
* `model`: The (string) representing the piscis model to be used. If using a custom model, this is the filename of that model.
* `spot_channel`: The name of the channel from the above list to be used to call spots.
* `expand_mask`: Distance in pixels to expand each mask label by. Useful when only DAPI mask is available.
##### Plotting
* `plot_max`: Boolean; Make an interactive plot of all dots on the max-projected image.
* `plot_z`: Boolean; Interactive plot of all dots on their respective Z-slices.
* `plot_out_dir`: Directory to output interactive plots to in `html` format.

A properly formatted `settings.yml` would look like:
```
image_dir: "/path/to/image.tif"              
image_type: "tif"             
spot_out: "./spots"   

channels:                   
 - "A647"
 - "DAPI"
 - "Cy3"
spot_channel: "A647"        
stack: true                   
piscis_thresh: 1              
piscis_scale: 1                
piscis_min_distance: 1
expand_mask: 0        

custom_model: true
model_dir: "/path/to/model/dir"
model: "20230905"         

plot_max: true             
plot_z: true                    
plot_out_dir: "."        
```
### Output
Upon successful completion, the pipeline will output 2 files per processed image file in the specified output directory `spot_out`. \
* Spots per cell
* Indiviual spot coordinates (x,y,z)
### Visualization
Two types of plots are generated if the user specifies so: a max-projected version of the image with all spots from all z-stack layers superimposed (`plot_max`), and a series of seperate z-stack images with their respective spots superimposed (`plot_z`). \
##### Max-projected plot
![image](https://github.com/user-attachments/assets/5daad508-fa36-4fc9-a32a-6f56a00aa165)

##### Z-stack plot
![image](https://github.com/user-attachments/assets/06de645e-aeb4-44ca-9fd8-a52336c84140)
