import numpy as np
import scipy
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
            left_patch = left_img[(row-r):(row+r), (col-r):(col+r)]
            right_strip = right_img[(row-r):(row+r), (col-r-max_disp):(col+r-min_disp)]

            # Transforming the patches into vectors so we can run them through pdist2
            lpvec = left_patch[:]
            rsvecs = np.zeros((patch_size^2, max_disp - min_disp + 1))
            for i in range(0,patch_size):
                print(rsvecs.shape)
                # TODO: Figure out indexing here
                rsvecs[((i-1)*patch_size+1):(i*patch_size), :] = right_strip[:, i:(max_disp - min_disp + i)]
            ssd = scipy.spatial.distance.pdist(disp_img,metric='euclidean')

    return disp_img
