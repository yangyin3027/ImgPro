import numpy as np
import skimage
from PIL import Image
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.ndimage as ndi

import os
import logging

class WhiteBalance:
    def percentile(self, img, percentile=0.05):
        if isinstance(percentile, float):
            if percentile < 0.5:
                percentile = (percentile, 1-percentile)
            else:
                percentile = (1-percentile, percentile)
        
        assert isinstance(percentile, (tuple, list))
        
        channels = [img[:,:, i] for i  in range(img.shape[-1])]
        new_chann = []
        for channel in channels:
            mi, ma = np.percentile(channel, percentile)
            channel = np.clip((channel-mi)/(ma-mi), 0, 1)
            new_chann.append(channel)
        return np.dstack(new_chann)
    
    def white_patch(self, img, patch):
        '''
        patch : (x, y, w, h) patch coordiante for white object in img
        ''' 
        if len(patch) == 4:
            x, y = patch[:2]
            w, h = patch[2:]

            rect = Rectangle((x, y), w, h,
                            linewidth=3, edgecolor='r',facecolor='none')
            img_patch = img[x:x+w, y:y+h, :]
            ref = img_patch.max(axis=(0,1))
        elif len(patch) == 2:
            x, y = patch
            rect = Rectangle((x, y), 10, 10,
                            linewidth=3, edgecolor='r',facecolor='none')
            ref = img[x, y, :]
        wb_img = img * 1.0 / ref
        wb_img = np.clip(wb_img, 0, 1)
        return wb_img, rect
    
    def auto_white_patch(self, img, radius=10):
        # preprocess the image first
        blured_img = ndi.gaussian_filter(img, sigma=1.5,
                                         radius=radius)
        # convert the image to gray
        blured_img = skimage.color.rgb2gray(blured_img)

        # find the brightest spot
        maxloc = np.argwhere(blured_img == blured_img.max())
        maxloc = np.squeeze(maxloc)

        return self.white_patch(img, maxloc)
    
    def gray_world(self, img):
        img = img * (img.mean()/img.mean(axis=(0,1)))
        img = img.clip(0, 1)
        return img

def img_to_uint(img):
    return (img*255).astype(np.uint8)

def main(file, method='patch', patch=None, 
         radius=10, percentile=None,
         plot=True, saved_folder='wb'):
    
    logger = logging.getLogger('whitebalance')
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(name)s -> %(message)s')
    sh.setFormatter(formatter)
    
    if os.path.isdir(file):
        files = [os.path.join(file, f)
                 for f in os.listdir(file)
                 if not f.startswith('.')]
    else:
        files = [file]

    wb = WhiteBalance()
    
    num_cols = min(len(files), 10)
    fig, axes = plt.subplots(2, num_cols, figsize=(20,10),
                                layout='constrained')
    for i, f in enumerate(files):
        img = np.array(Image.open(f))
        # convert img to [0, 1] float
        img = img.astype(float) / float(img.max())

        if method == 'patch':
            if not patch:
                wb_img, rect = wb.auto_white_patch(img, radius=radius)
            else:
                wb_img, rect = wb.white_patch(img, patch)
        elif method == 'percentile':
            if not percentile:
                percentile = 0.01
            wb_img = wb.percentile(img, percentile)
            rect = None
        else:
            wb_img = wb.gray_world(img)
            rect = None
        
        logger.info(f'{method} used, {i+1} images processed!')
        
        wb_img = img_to_uint(wb_img)

        if saved_folder:
            os.makedirs(saved_folder, exist_ok=True)
            saved_fname = f.split('/')[-1].split('.')[0] + '_wb.tif'
            Image.fromarray(wb_img).save(os.path.join('.', saved_folder, saved_fname))
            logger.info(f'image {saved_fname} saved!')

        if i < num_cols:
            axes[0, i].imshow(img_to_uint(img))
            if rect:
                axes[0, i].add_patch(rect)
            axes[1, i].imshow(wb_img)

    for a in axes.ravel():
        a.axis("off")
    plt.show()
       
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-m', '--mode', type=str, default='patch')
    parser.add_argument('--percentile', default=0.01)
    parser.add_argument('-s', '--save', type=str, default=None)
    parser.add_argument('-p', '--patch', nargs='+',type=int,
                        default=None)
    parser.add_argument('-r', '--radius', type=int, default=10)

    args = parser.parse_args()

    main(args.file,
         method=args.mode,
         percentile=args.percentile,
         saved_folder=args.save,
         radius=args.radius,
         patch=args.patch)
