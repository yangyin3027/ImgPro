import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def white_balance(img, perc = 0.05):
    channels = [channel for channel in cv2.split(img)]
    new_chann = []
    for channel in channels:
        mi, ma = np.percentile(channel, perc), np.percentile(channel, 100.0-perc)
        channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
        new_chann.append(channel)
    return np.dstack(new_chann)

def read_wb_save(data_dir, suffix = '_wb.tif'):
    files = [f for f in os.listdir(data_dir) if f.endswith('tif')]
    imgs = [cv2.imread(os.path.join(data_dir, file), 1) for file in files]
    imgs_wb = [white_balance(img) for img in imgs]
    for i, img in enumerate(imgs_wb):
      cv2.imwrite(os.path.join(data_dir, files[i].split('.')[0] + suffix), img)

def imshow_wb(data_dir, n = 2, suffix = '_wb.tif', gap = 20):
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
    imgs_file = [f for f in files if not f.endswith(suffix)]
    imgs_file.sort()
    imgs_wb_file = [f for f in files if f.endswith(suffix)]
    imgs_wb_file.sort()

    imgs = [cv2.cvtColor(cv2.imread(os.path.join(data_dir,img), 1), cv2.COLOR_BGR2RGB)
            for img in imgs_file]
    imgs_wb = [cv2.cvtColor(cv2.imread(os.path.join(data_dir,img), 1), cv2.COLOR_BGR2RGB)
            for img in imgs_wb_file]

    n = min(len(imgs), n)
    blank_x = np.full((imgs[0].shape[0], gap, imgs[0].shape[2]), 255)
    blank_y = np.full((gap, imgs[0].shape[1]*2 + gap, imgs[0].shape[2]), 255)

    big_img = []
    for i in range(n):
      new_img = np.concatenate((imgs[i], blank_x, imgs_wb[i]), axis = 1)
      big_img.append(np.concatenate((new_img, blank_y), axis = 0))

    final_img = np.uint8(np.concatenate(big_img, axis = 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(final_img)
    plt.axis('off')
    plt.show()

if __name__ = '__main__':
    read_wb_save(data_dir)
    imshow_wb(data_dir)

    
    

