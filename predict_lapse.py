from predict import*
import os
import cv2
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Visualize predictions at each epoch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', '-f', help='path to model folder', type=str, required=True)

    parser.add_argument('--video-name', '-n', type=str, required=True, help='Name of new movie')

    parser.add_argument('--save-im', '-s', action='store_true', default=False, help='Save the predictions in a folder(y/n)')

    return parser.parse_args()


def pred_lps(folder_name, video_name, save_im):
    folder_name = "UnetPlusPlus_0.5_resnet18_imagenet_wogal07_predict" #Folder with saved models
    video_name = folder_name + ".avi"
    folder = os.listdir(folder_name)



    def file_sort(folder):
        pth_fold = []
        for ff in folder:
            if not ff.startswith('events'):
                pth_fold.append(ff)

        num_folder = []
        fl_pref = None
        for f in pth_fold: # Remove .pth from every file
            fl_pref = f[0:8]
            fl = f[:-4]
            num = fl[8:]
            num_folder.append(num)


        int_ls = [int(i) for i in num_folder]
        int_ls_sort = sorted(int_ls)
        #str_ls_sort = [str(j) for j in int_ls_sort]

        folder = [fl_pref + str(k) + ".pth" for k in int_ls_sort]

        return folder


    folder_sort = file_sort(folder)

    os.mkdir("temp")

    image_nms = []
    for file in folder_sort: #Generate and save the predictions
        model_pth = os.path.join(folder_name, file)
        fl = file[:-4]
        epoch_num = fl[8:]
        fl_nm = "temp/Gallagher_071417_Images_T=0001" + "_epoch" + str(epoch_num) + '.tif'
        image_nms.append(fl_nm)
        cmd = "python predict.py -m " +  model_pth + " -i Gal_071417_raw/Gallagher_071417_Images_T=0117.tif -o " + fl_nm
        os.system(cmd)


    image_folder = 'temp'

    images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    if not save_im:
        shutil.rmtree("temp")


if __name__ == '__main__':
    args = get_args()
    folder_name=args.folder
    video_name=args.video_name
    save_im = args.save_im
    pred = pred_lps(folder_name, video_name, save_im)
