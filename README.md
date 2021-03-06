# Time-Lapse-Microscopy-Toolkit

This toolkit provides: 
  * Image segmentation 
    * Five CNN architectures
    * Thirteen encoders with pre-trained weights
    * Augmentation 
  * Cell counter
  * Cell tracking  
  * No coding required 
- - - - 

  ## Example ##

``` shell script
# Create a static library of augmented images
> python augmentation.py -imf images -msf masks -a 4 -imfn train_images -mskfn train_masks -s 768 -gs True

# Train the model with the augmented and original images. The UNet++ architecture and inceptionv4 encoder resulted in the most accurate segmentation according to our tests.
> python train.py -e 200 -b 4 -cp Segmentation_test/ -fn train_images/ -mf train_masks/ -en resnet18 -wt imagenet -a unetplusplus

# Monitor the training using Tensorboard
> python tensorboard --logdir=Segmentation_test

# Use predict lapse to determine which epoch produced the best results
> python predict_lapse.py -f Segmentation_test -n test_folder -en resnet18 -wt imagenet -a unetplusplus

# Make predictions using the trained model 
> python predict.py -m Segmentation_test/CP_epoch11.pth -i images/ -t 0.1 -en resnet18 -wt imagenet -a unetplusplus

# Move the predictions from the source folder to a new folder (e.g., predictions)
> python cell_tracking.py -f predictions


```

## Segmentation  ##

This package uses the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch "Segmentation Models Pytorch") package to provide a wide range of CNN architectures and encoders for image segmentation. 

#### Architectures ####

* Unet
* UnetPlusPlus
* MAnet
* Linknet
* FPN

#### Encoders ####

* See full list of encoders on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch "Segmentation Models Pytorch")

<p align="center">
  <img width="586" alt="Screen Shot 2021-05-30 at 2 57 44 PM" src="https://user-images.githubusercontent.com/58287074/120590926-26798e80-c3f0-11eb-82bd-7fd6b4d06903.png">
</p>

## Cell Counting and Tracking ##
  
After segmentation, this package uses the [Trackpy](https://github.com/soft-matter/trackpy "Trackpy") package for cell counting and tracking. When cell_tracking.py is run, the program displays the centroids of a sample image, the total movement of each cell, prints the mean number of cells per frame, the total number of tracks, the mean track distance, and returns a Pandas data structure with raw data on every frame. 

<p float="center">
  <img width="400" height="300"alt="Screen Shot 2021-06-02 at 10 43 23 PM" src="https://user-images.githubusercontent.com/58287074/120593437-5cb90d00-c3f4-11eb-8635-3972614ebdfc.png">
  <img width="400" alt="Screen Shot 2021-06-02 at 10 44 31 PM" src="https://user-images.githubusercontent.com/58287074/120593632-a0ac1200-c3f4-11eb-8c3d-807d148641b8.png">
</p>


### Augmentation ###

``` shell script

> python augmentation.py -h
usage: augmentation.py [-h] [--image-folder IMAGE_FOLDER]
                       [--mask-folder MASK_FOLDER] [--aug-size AUG_SIZE]
                       [--im-folder-nm IM_FOLDER_NM]
                       [--msk-folder-nm MSK_FOLDER_NM] [--scale SCALE]
                       [--grayscale]

Create a new augmented dataset

optional arguments:
  -h, --help            show this help message and exit
  --image-folder IMAGE_FOLDER, -imf IMAGE_FOLDER
                        Path to image folder (default: None)
  --mask-folder MASK_FOLDER, -msf MASK_FOLDER
                        Path to mask folder (default: None)
  --aug-size AUG_SIZE, -a AUG_SIZE
                        How many times to augment the original image folder
                        (default: None)
  --im-folder-nm IM_FOLDER_NM, -imfn IM_FOLDER_NM
                        Name for new augmented image folder (default: None)
  --msk-folder-nm MSK_FOLDER_NM, -mskfn MSK_FOLDER_NM
                        Name for new augmented mask folder (default: None)
  --scale SCALE, -s SCALE
                        Dimension to scale ass the images (default: 768)
  --grayscale, -gs      Make all the augmented images grayscale (default:
                        False)
```

### Training ###

```shell script

> python train.py -h     
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL] [--classes CLASSES] [--in-channels IN_CHANNELS] [--device DEVICE]
                [-cp CHECKPOINT] [-fn FILE] [-en ENCODER] [-wt WEIGHT]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100) (default: 10.0)
  --classes CLASSES, -c CLASSES
                        Model output channels (default: 1)
  --in-channels IN_CHANNELS, -ic IN_CHANNELS
                        Model input channels (default: 1)
  --device DEVICE, -d DEVICE
                        Select device (default: cuda:0)
  -cp CHECKPOINT, --checkpoint CHECKPOINT
                        Name folder for checkpoints (default: checkpoints/)
  -fn FILE, --file FILE
                        Name folder for images (default: None)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
                                              
```

### Predict Lapse ###

``` shell script 

> python predict_lapse.py 
usage: predict_lapse.py [-h] --folder FOLDER [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE] --input INPUT [INPUT ...] [-n NAME]
predict_lapse.py: error: the following arguments are required: --folder/-f, --input/-i
nathanburg@Nathans-MBP CNN-Architecture-Comparison- % python3 predict_lapse.py -h
usage: predict_lapse.py [-h] --folder FOLDER [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE] --input INPUT [INPUT ...] [-n NAME]

Visualize predictions at each epoch

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER, -f FOLDER
                        path to model folder (default: None)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Name of architecture (default: None)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  -n NAME, --name NAME  Name for image folder (default: None)
```

### Prediction ###

``` shell script

> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] [--output INPUT [INPUT ...]] [--viz] [--no-save] [--mask-threshold MASK_THRESHOLD]
                  [--scale SCALE] [--classes CLASSES] [--in-channels IN_CHANNELS] [--device DEVICE] [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default: False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white (default: None)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
  --classes CLASSES, -c CLASSES
                        Model output channels (default: 1)
  --in-channels IN_CHANNELS, -ic IN_CHANNELS
                        Model input channels (default: 1)
  --device DEVICE, -d DEVICE
                        Select device (default: cuda:0)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Name of architecture (default: None)
                        
  ```
  
  
  ### Cell Counting and Tracking ###
  
  ``` shell script
> python cell_tracking.py -h
usage: cell_tracking.py [-h] --folder FOLDER

Count and track cells

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER, -f FOLDER
                        path to image folder (default: None)
                        
```

## Installation ##
Create a folder called Time_Lapse_Microscopy_Toolkit.

``` shell script

$ git clone https://github.com/nathanBurg/Time-Lapse-Microscopy-Toolkit.git Time_Lapse_Microscopy_Toolkit
$ pip3 install -r /path/to/Time_Lapse_Microscopy_Toolkit/requirements.txt 

```

  

  
