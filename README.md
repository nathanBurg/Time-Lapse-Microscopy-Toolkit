# Time-Lapse-Microscopy-Toolkit

This toolkit provides: 
  * Image segmentation 
    * Six CNN architectures
    * Thirteen encoders with pre-trained weights
    * Augmentation 
  * Cell counter
  * Cell tracking  
  * No coding required 
- - - - 
## Segmentation  ##

This package uses the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch "Segmentation Models Pytorch") package to provide a wide range of CNN architectures and encoders for image segmentation. 

### Training ###

```shell script

> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]
                [--classes CLASSES] [--in-channels IN_CHANNELS]
                [--device DEVICE] [-cp CHECKPOINT] [-fn FILE]
                [-mf MASK_FOLDER] [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE]

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
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)
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
  -mf MASK_FOLDER, --mask-folder MASK_FOLDER
                        Name for folder for mask (default: None)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Name of architecture (default: None)
```


### Prediction ###

``` shell script

C:\Users\kevin\Desktop\final_cnn>python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]
                  [--classes CLASSES] [--in-channels IN_CHANNELS]
                  [--device DEVICE] [-en ENCODER] [-wt WEIGHT]
                  [-a ARCHITECTURE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: None)
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
  
  ## Cell Counting and Tracking ##
  
  After segmentation, this package provides cell counting and cell tracking. 
  
  
  
