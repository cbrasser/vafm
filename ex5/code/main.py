import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ex5_helpers import get_disparity

# Scaling down by a factor of 2, otherwise too slow.
left_img = Image.open('../data/left/000000.png').convert('L')
right_img = Image.open('../data/right/000000.png').convert('L')
height = np.shape(left_img)[1]
width = np.shape(left_img)[0]
left_img = left_img.resize((int(0.5*height),int(0.5*width)))
right_img = left_img.resize((int(0.5*height),int(0.5*width)))
left_img = np.asarray(left_img)
right_img = np.asarray(right_img)
K = np.loadtxt('../data/K.txt')
K[1:2, :] = K[1:2, :] / 2

poses = np.loadtxt('../data/poses.txt')

# Given by the KITTI dataset:
baseline = 0.54

# Carefully tuned by the TAs:
patch_radius = 5
min_disp = 5
max_disp = 50
xlims = [7,20]
ylims = [-6, 10]
zlims = [-5, 5]


disp_img = get_disparity(left_img, right_img, patch_radius, min_disp, max_disp)
plt.matshow(disp_img)
plt.show()
