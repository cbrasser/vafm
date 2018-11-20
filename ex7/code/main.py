import os, random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from modules import parabola_ransac, ransac_localization


back = '..'
data_dir = 'data'
plot_parabola_ransac = False
#np.random.seed(1)

num_inliers = 20
num_outliers = 10
noise_ratio = 0.1
poly = np.random.rand(3,1)
extremum = -np.poly(2)/(2*np.poly(1))
extremum = extremum[0]
xstart = extremum -0.5
lowest = np.polyval(poly, extremum)
highest = np.polyval(poly, xstart)
xspan = 1
yspan = highest - lowest
#max_noise = noise_ratio * yspan
max_noise = 0.02

x = np.random.rand(1, num_inliers) + xstart
y = np.polyval(poly, x)

data = np.vstack((np.append(x, np.random.rand(1, num_outliers) + xstart),
 np.append(y, np.random.rand(1, num_outliers)*yspan + lowest)))


# Config files

K = np.loadtxt(os.path.join(back, data_dir, 'K.txt'))
keypoints = np.loadtxt(os.path.join(back, data_dir,'keypoints.txt'))
p_W_landmarks = np.loadtxt(os.path.join(back, data_dir,'p_W_landmarks.txt'))

database_image = Image.open(os.path.join(back, data_dir, '000000.png')).convert('L')

# -------------------------------------------------------Part 1-------------------------------------------------------

best_guess_history, max_num_inliers_history = parabola_ransac(data, max_noise)
if plot_parabola_ransac:
    print(f'Got {len(best_guess_history)} fits')
    plt.scatter(data[0],data[1])
    x = np.arange(start= xstart, stop= xstart+1, step=0.01)
    #Truth
    plt.plot(x, np.polyval(poly, x),'b', linewidth = 5)
    #Guesses
    for i in range(len(best_guess_history)-1):
        plt.plot(x,np.polyval(best_guess_history[i], x),'y')
    #Best guess
    plt.plot(x,np.polyval(best_guess_history[len(best_guess_history)-1], x),'r',linewidth=3)
    plt.show()

# ------------------------------------------------------- Parts 2 & 3 -------------------------------------------------------

query_image = Image.open(os.path.join(back, data_dir,'000001.png')).convert('L')
R_C_W, t_C_W, all_matches, inlier_mask, max_num_inliers_history = ransac_localization(query_image, database_image, keypoints, p_W_landmarks, K)
