import numpy as np
import os
import argparse
import glob
import skimage.measure as measure
from scipy.misc import imread
from PIL import Image
import ssim
import cv2


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_ROOT_FMT = os.path.abspath(os.path.join(SCRIPT_DIR, 'results', 'MNIST', 'images', 'data=%s', 'model=%s'))
QUANT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, 'results', 'MNIST', 'quantitative'))
QUANT_FILE_PATH_FMT = os.path.abspath(os.path.join(QUANT_ROOT, 'data=%s', 'model=%s', 'results.npz'))

def to_pil_image(a):
    '''
    Convert NumPy array a into a PIL image
    :param a: np.uint8 array with values in range [0, 255]
    :return: Image instance corresponding to m
    '''
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))

def compute_metrics_single_video(video_root, K, T):
    '''
    Compute frame-wise SSIM and PSNR for the GT and prediction frames at the given folder
    :param video_root: Path of the directory containing the `raw_gt` and `raw_pred` images
    :return:
    '''
    files = os.listdir(video_root)
    gt_files = sorted(filter(lambda x: x.startswith('raw_gt'), files))
    pred_files = sorted(filter(lambda x: x.startswith('raw_pred'), files))
    assert(len(gt_files) == len(pred_files))

    psnr_arr = np.zeros(T)
    ssim_arr = np.zeros(T)
    for i in xrange(K, K+T):
        gt_image = imread(os.path.join(video_root, gt_files[i]))
        pred_image = imread(os.path.join(video_root, pred_files[i]))
        psnr_arr[i-K] = measure.compare_psnr(gt_image, pred_image)
        ssim_arr[i-K] = ssim.compute_ssim(to_pil_image(gt_image), to_pil_image(pred_image))

    return psnr_arr, ssim_arr


def main(modelSliceName, dataSliceName, numPastFrames, numFutureFrames):
    # Set up file paths
    image_root = IMAGE_ROOT_FMT % (dataSliceName, modelSliceName)
    quant_file = QUANT_FILE_PATH_FMT % (dataSliceName, modelSliceName)
    if not os.path.isdir(os.path.dirname(quant_file)):
        os.makedirs(os.path.dirname(quant_file))
    image_folders = filter(lambda x: not x.endswith('.gif'), os.listdir(image_root))

    # Evaluate each video
    psnr_rows = []
    ssim_rows = []
    for folder in sorted(image_folders):
        psnr_arr, ssim_arr = compute_metrics_single_video(os.path.join(image_root, folder), numPastFrames, numFutureFrames)
        psnr_rows.append(psnr_arr)
        ssim_rows.append(ssim_arr)

    # Save error matrices
    psnr_err = np.stack(psnr_rows, axis=0)
    ssim_err = np.stack(ssim_rows, axis=0)
    np.savez(quant_file, psnr=psnr_err, ssim=ssim_err)
    print('Done with data=%s, model=%s' % (dataSliceName, modelSliceName))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelSliceName', type=str, required=True,
                        help='Name of the new MNIST slice the desired model was trained on')
    parser.add_argument('--dataSliceName', type=str, required=True,
                        help='Name of the new MNIST slice to evaluate on')
    parser.add_argument('--numPastFrames', type=int, default=10,
                        help='Number of frames to condition on')
    parser.add_argument('--numFutureFrames', type=int, default=5,
                        help='Number of frames to predict')
    args = parser.parse_args()
    main(**vars(args))