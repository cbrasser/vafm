import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os
from PIL import Image
import util

DATA = '../data'
I_R = np.asarray(Image.open(os.path.join(DATA, '000000.png')).convert('L'))
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
