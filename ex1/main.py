import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ex1_helpers import distort_points, positionVec2TransformMatrix, project_points, distort_points, undistortImageVectorized

square_size = 0.04

num_corners_x = 9
num_corners_y = 6
num_corners = num_corners_x*num_corners_y
lineWidth = 4
pointRadius = 90

cubeSize = 0.04

# X, Y = np.meshgrid(np.arange(0,num_corners_x-1),np.arange(0,num_corners_y-1))
# X = X * square_size
# Y = Y* square_size
p_W_corners = [[square_size*x,square_size*y] for x in range(0,num_corners_x) for y in range(0,num_corners_y)]
p_W_corners = np.transpose(np.array(p_W_corners))
z_dim = np.zeros(num_corners)
p_W_corners = np.vstack([p_W_corners, z_dim])
poses = np.loadtxt('data/poses.txt')
K = np.loadtxt('data/K.txt')
D = np.loadtxt('data/D.txt')



img_index = 1

img = Image.open('data/images/img_000{}.jpg'.format(img_index)).convert('L')
# img.save('greyscale.png')
img = np.asarray(img)
print('image shape: ',img.shape)
T_C_W = positionVec2TransformMatrix(poses[img_index,:])
temp = np.vstack([p_W_corners, np.ones((1,num_corners))])
p_C_corners = np.matmul(T_C_W, temp)
p_C_corners = p_C_corners[0:3,:]

projected_pts = project_points(p_C_corners, K, D);


# From world to camera, in oder to project it onto the image plane afterwards


img_undistorted_vectorized = undistortImageVectorized(img,K,D);

# figure()
# plt.imshow(img_undistorted_vectorized)
plt.imshow(img)
plt.scatter(projected_pts[0,:], projected_pts[1,:])
plt.scatter(projected_pts[0,:], projected_pts[1,:])
# From world to camera, in oder to project it onto the image plane afterwards
plt.show()
