import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os
from PIL import Image
import util

DATA = '../data'
I_R = np.asarray(Image.open(os.path.join(DATA, '000000.png')).convert('L'))
def part_1():
    plt.subplot(2,2,1)
    plt.imshow(I_R)
    plt.title('Reference')

    plt.subplot(2,2,2)
    W = util.get_sim_warp(50, -30, 0, 1)
    plt.imshow(util.warp_image(I_R, W))
    plt.title('Translation')

    plt.subplot(2,2,3)
    W = util.get_sim_warp(0, 0, 10, 1);
    plt.imshow(util.warp_image(I_R, W))
    plt.title('Rotation around upper left corner')

    plt.subplot(2,2,4)
    W = util.get_sim_warp(0, 0, 0, 0.5);
    plt.imshow(util.warp_image(I_R, W))
    plt.title('Zoom on upper left corner')

    plt.show()

#----------------------------Part 2--------------------------
def part_2():
    # Get and display template:
    plt.subplot(1, 2, 1)
    W0 = util.get_sim_warp(0, 0, 0, 1)
    x_T = np.asarray([900, 291])
    r_T = 15
    template = util.get_warped_patch(I_R, W0, x_T, r_T)
    plt.imshow(template)
    # axis equal;
    # axis off;
    plt.title('Template')

    plt.subplot(1, 2, 2)
    W = util.get_sim_warp(10, 6, 0, 1)
    I = util.warp_image(I_R, W)
    r_D = 20
    dx, ssds = util.track_brute_force(I_R, I, x_T, r_T, r_D)
    plt.imshow(ssds)
    # axis equal;
    # axis off;
    plt.title('SSDs')
    print([f'Displacement best explained by (dx, dy) = ( {dx} )'])
    plt.show()

part_2()
