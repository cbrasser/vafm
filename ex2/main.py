import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


img_index = 1

undimg_path = 'data/images_undistorted/img_0{}.jpg'.format(img_index)
undimg = np.asarray(Image.open(undimg_path).convert('L'))

K = np.loadtxt('data/K.txt');

p_W_corners = 0.01 * np.loadtxt('data/p_W_corners.txt')
num_corners = len(p_W_corners)

# Load the 2D projected points (detected on the undistorted image)
all_pts2d = np.loadtxt('data/detected_corners.txt')
pts2d = all_pts2d[img_index,:]
pts2d = np.reshape(pts2d, (2, 12))


M_dlt = estimate_pose_DLT(pts2d, p_W_corners, K)
