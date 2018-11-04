import numpy as np
from PIL import Image



square_size = 0.04

num_corners_x = 9
num_corners_y = 6
num_corners = num_corners_x*num_corners_y
lineWidth = 4
pointRadius = 90

cubeSize = 0.04

X, Y = np.meshgrid(np.arange(0,num_corners_x-1),np.arange(0,num_corners_y-1))
X = X * square_size
Y = Y* square_size
p_W_corners = [[square_size*x,square_size*y] for x in range(0,num_corners_x) for y in range(0,num_corners_y)]
p_W_corners = np.transpose(np.array(p_W_corners))
z_dim = np.zeros(num_corners)
p_W_corners = np.vstack([p_W_corners, z_dim])
print(p_W_corners.shape)
poses = np.loadtxt('data/poses.txt')
K = np.loadtxt('data/K.txt')
D = np.loadtxt('data/D.txt')

img_index = 1

img = Image.open('data/images/img_000{}.jpg'.format(img_index)).convert('LA')
img.save('greyscale.png')

#Left here

k1 = D(1)
k2 = D(2)

cubeX = 7
cubeY = 5

firstRotation = poses[1:1, 1:3]
firstTranslation = poses[1:1, 4:6]

#From world to camera, in oder to project it onto the image plane afterwards



def positionVec2TransformMatrix(positions):
    omega = positions[1:3]
    t = positions[4:6]

    theta = np.norm(omega)
    k = omega/theta
    kx = k[1]
    ky = k[2]
    kz = k[3]
    K = np.matrix([0, -kz, ky],[kz, 0, -kx],[-ky, kx, 0])

    R = np.identity(3) + np.sin(theta) * K + (1-np.cos(theta)) * np.pow(K,2);

    T = np.identity(4);
    T[1:3,1:3] = R;
    T[1:3,4] = t;



def project_points(points, K, D = np.zeros(4,1)):
    #Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    #distortion coefficients (4x1).


    num_points = np.shape(points)[1]

    #get normalized coordinates
    xp = np.divide(points[1,:],points[3,:])
    yp = np.divide(points[2,:],points[3,:])

    #apply distortion
    x_d = distortPoints(np.matrix(xp,yp),D)
    xpp = x_d[1,:]
    ypp = x_d[2,:]

    #convert to pixel coordinates
    projected_points = K * np.matrix(xpp,ypp,np.ones(1, num_points))
    projected_points = projected_points[1:2, :];
    return project_points

def distort_points(x,D):
    k1 = D[1]
    k2 = D[2]

    xp = x[1,:]
    yp = x[2,:]

    r2 = np.power(xp,2) + np.power(yp,2)
    xpp = np.multiply(xp, 1 + k1*r2 + k2*np.power(r2,2))
    ypp = np.multiply(yp, 1 + k1*r2 + k2*np.power(r2,2))


    x_d = np.matrix(xpp,ypp)
    return x_d
