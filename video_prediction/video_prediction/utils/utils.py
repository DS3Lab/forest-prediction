import numpy as np

def rescale_if_negative(img):
    # might have negative values
    print('RESCALE IF NEGATIVE', img.shape)
    if len(img.shape == 4): # list of images
        rescaled_images = np.zeros(img.shape)
        for i in range(img.shape[0])
            if img[i].min() < 0:
                rescaled_images[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
            else:
                rescaled_images[i] = img[i]
        return rescaled_images
    else: # only one image
        if img.min() < 0:
            return (img - img.min()) / (img.max() - img.min())
        else:
            return img
