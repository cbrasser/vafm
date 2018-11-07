import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image

def get_disparity(left_img, right_img, patch_radius, min_disp, max_disp):
    '''
        left_img and right_img are both H x W and you should return a H x W
        matrix containing the disparity d for each pixel of left_img. Set
        disp_img to 0 for pixels where the SSD and/or d is not defined, and for d
        estimates rejected in Part 2. patch_radius specifies the SSD patch and
        each valid d should satisfy min_disp <= d <= max_disp.
    '''
    disp_img = np.zeros(np.shape(left_img))
    patch_size = 2 * patch_radius + 1
    r = patch_radius
    width = np.shape(left_img)[0]
    height = np.shape(left_img)[1]

    for row in range(patch_radius,width-patch_radius):
        for col in range(max_disp + patch_radius,height - patch_radius):
            left_patch = left_img[(row-r):(row+r+1), (col-r):(col+r+1)]
            right_strip = right_img[(row-r):(row+r+1), (col-r-max_disp):(col+r-min_disp+1)]


            left_patch_vec = left_patch.reshape((121,1))
            right_strip_vec = np.zeros((np.power(patch_size,2), max_disp - min_disp + 1))
            for i in range(0,patch_size):
                right_strip_vec[((i)*patch_size):((i+1)*patch_size), :] = right_strip[:, i:(max_disp - min_disp + i + 1)]
            ssds = cdist(np.transpose(left_patch_vec),np.transpose(right_strip_vec),metric='sqeuclidean')
            min_ssd = np.argmin(ssds)
            disp_img[row, col] = max_disp - min_ssd

    return disp_img
