import numpy as np

def distort_points(x,D):
    k1 = D[0]
    k2 = D[1]

    xp = x[0,:]
    yp = x[1,:]

    r2 = np.power(xp,2) + np.power(yp,2)
    xpp = np.multiply(xp, 1 + k1*r2 + k2*np.power(r2,2))
    ypp = np.multiply(yp, 1 + k1*r2 + k2*np.power(r2,2))


    x_d = np.vstack([xpp,ypp])
    return x_d

def positionVec2TransformMatrix(positions):
    omega = positions[0:3]
    t = positions[3:6]

    theta = np.linalg.norm(omega)
    k = omega/theta
    kx = k[0]
    ky = k[1]
    kz = k[2]
    K = np.array([[0, -kz, ky],[kz, 0, -kx],[-ky, kx, 0]])

    R = np.identity(3) + np.sin(theta) * K + (1-np.cos(theta)) * np.power(K,2)
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def project_points(points, K, D = np.zeros((4,1))):
    #Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    #distortion coefficients (4x1).

    num_points = np.shape(points)[1]

    #get normalized coordinates
    xp = np.divide(points[0,:],points[2,:])
    yp = np.divide(points[1,:],points[2,:])

    #apply distortion
    x_d = distort_points(np.vstack([xp,yp]),D)
    xpp = x_d[0,:]
    ypp = x_d[1,:]

    #convert to pixel coordinates
    projected_points = np.matmul(K,np.vstack([xpp,ypp,np.ones((1, num_points))]))
    projected_points = projected_points[0:2, :];
    print(projected_points.shape)
    return projected_points

def undistortImageVectorized(img, K, D):
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]
    # X_, Y_ = np.meshgrid(np.arange(0,np.shape(img)[1]),np.arange(0,np.shape(img)[0]))
    # nonzeros = (X!=0).sum()
    px_locs = []
    for x in range(img_width):
        for y in range(img_height):
            px_locs.append([x,y,1])
    # px_locs = [[x,y] for x in range(0,img_width) for y in range(0,img_height)]
    # px_locs = np.hstack([X_[:],Y_[:],np.ones((nonzeros,1))])
    px_locs = np.transpose(np.array(px_locs))
    print(px_locs.shape)
    normalized_px_locs = np.matmul(np.power(K,-1),px_locs)
    normalized_px_locs = normalized_px_locs[0:2, :]
    normalized_dist_px_locs = distort_points(normalized_px_locs, D)
    dist_px_locs = np.matmul(K,np.vstack([normalized_dist_px_locs,
    np.ones((1,np.shape(normalized_dist_px_locs)[1]))]))
    dist_px_locs = dist_px_locs[0:2, :]
    #dist_px_locs is nan
    intensity_vals = img[np.round(dist_px_locs[1, :]) + np.shape(img)[0] * np.round(dist_px_locs[1, :])]
    undimg = np.uint8(np.reshape(intensity_vals, np.shape(img)))
