from predict import*
import os
import cv2
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Visualize predictions at each epoch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', '-f', help='path to model folder', type=str, required=True)

    parser.add_argument('-en', '--encoder', type=str,
                        help="Name of encoder",
                        default='resnet34')
    parser.add_argument('-wt', '--weight', type=str,
                        help="Encoder weights")
    parser.add_argument('-a', '--architecture', type=str,
                        help="Name of architecture")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('-n', '--name', type=str,
                        help="Name for image folder")

    return parser.parse_args()


def pred_lps(folder_name, encoder, weight, architecture, in_file, name):

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
            if num != '':
                num_folder.append(num)

        int_ls = [int(i) for i in num_folder]
        int_ls_sort = sorted(int_ls)
        #str_ls_sort = [str(j) for j in int_ls_sort]

        folder = [fl_pref + str(k) + ".pth" for k in int_ls_sort]

        return folder


    folder_sort = file_sort(folder)

    os.mkdir(name)

    image_nms = []
    for file in folder_sort: #Generate and save the predictions
        model_pth = os.path.join(folder_name, file)
        fl = file[:-4]
        epoch_num = fl[8:]
        spl_fl = file.split('/')[-1]

        fl_nm = name + "/epoch_" + epoch_num + '.tif'
        image_nms.append(fl_nm)
        cmd = "python3 predict.py -m " + str(model_pth) + " -i " + str(in_file[0]) + ' -en ' + str(encoder) + ' -a ' + str(architecture) + " -o " + str(fl_nm) + ' -wt ' + str(weight)
        os.system(cmd)



if __name__ == '__main__':
    args = get_args()
    folder_name=args.folder
    encoder = args.encoder
    weight = args.weight
    architecture = args.architecture
    in_file = args.input
    name = args.name
    pred = pred_lps(folder_name, encoder, weight, architecture, in_file, name)
