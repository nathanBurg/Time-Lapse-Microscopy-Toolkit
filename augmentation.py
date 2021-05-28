from aug_fxns import *
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
import argparse



# This function opens the all the images in a folder
def open_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def get_args():
    parser = argparse.ArgumentParser(description='Create a new augmented dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-folder', '-imf', type=str,
                        help="Path to image folder")

    parser.add_argument('--mask-folder', '-msf', type=str,
                        help="Path to mask folder")

    parser.add_argument('--aug-size', '-a', type=int,
                        help="How many times to augment the original image folder")

    parser.add_argument('--im-folder-nm', '-imfn', type=str,
                        help="Name for new augmented image folder")

    parser.add_argument('--msk-folder-nm', '-mskfn', type=str,
                        help="Name for new augmented mask folder")

    parser.add_argument('--scale', '-s', type=int, default=768,
                        help="Dimension to scale ass the images")

    parser.add_argument('--grayscale', '-gs', action='store_true', default=False,
                        help='Make all the augmented images grayscale')


    return parser.parse_args()


#Function to save augmentation to image/mask pair
def save_aug(img, msk, im_nm, msk_nm, im_folder_nm, msk_folder_nm):


    #Create separate folders for images and masks
    if os.path.isdir(im_folder_nm) == True and os.path.isdir(msk_folder_nm) == True:
        pass
    else:
        os.mkdir(im_folder_nm)
        os.mkdir(msk_folder_nm)

    augmentation = get_training_augmentation()
    sample = augmentation(image=img, mask=msk)
    image, mask = sample['image'], sample['mask']

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    if scale:
        image = image.resize((scale, scale))
        mask = mask.resize((scale, scale))

    if grayscale:
        image = ImageOps.grayscale(image)
        mask = ImageOps.grayscale(mask)

    image.save(im_folder_nm + '/' + str(im_nm) + '.tif')
    mask.save(msk_folder_nm + '/' + str(msk_nm) + '_mask.tif')


def aug_set(im_folder, msk_folder, aug_size, im_folder_nm, msk_folder_nm, scale, grayscale):

    try:
        im_folder = open_folder(im_folder)
        msk_folder = open_folder(msk_folder)
    except AssertionError as error:
        print(error)
        quit()

    count = 0

    for i in range(aug_size):
        for im, msk in zip(im_folder, msk_folder):
            count += 1
            nm = str(count)
            while len(nm) <= 3:
                nm = '0' + nm

            save_aug(img=im, msk=msk, im_nm=nm, msk_nm=nm, im_folder_nm=im_folder_nm, msk_folder_nm=msk_folder_nm)
            print('Saved image pair ' + nm)


if __name__ == '__main__':
    args = get_args()

    im_folder = args.image_folder
    msk_folder = args.mask_folder
    aug_size = args.aug_size
    im_folder_nm = args.im_folder_nm
    msk_folder_nm = args.msk_folder_nm
    scale = args.scale
    grayscale = args.grayscale

    aug = aug_set(im_folder=im_folder, msk_folder =msk_folder, aug_size=aug_size, im_folder_nm=im_folder_nm, msk_folder_nm=msk_folder_nm)
