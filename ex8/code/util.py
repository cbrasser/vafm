import numpy as np
from scipy.spatial.distance import cdist


def get_sim_warp(dx, dy, alpha_deg, lam):
    '''
    alpha in degree,
    '''
    alpha_rad = np.radians(alpha_deg)
    W = lam *np.array([(np.cos(alpha_rad),np.sin(alpha_rad), dx),
                    (-np.sin(alpha_rad),np.cos(alpha_rad), dy) ])
    return W


def warp_image(I_R, W):

    height, width = I_R.shape
    print(W)

    I = np.zeros((height, width))

    for x in range(0,width):
        for y in range(0, height):
            warp =W.dot(np.transpose(np.array([x,y,1])))
            warp = warp.astype(int)

            if warp[0] < 0  or warp[0] >= width or warp[1] < 0 or warp[1] >= height:
                I[y,x] = 0
            else:
                I[y,x] = I_R[warp[1],warp[0]]



    return I

def get_warped_patch(I, W, x_T, r_T):
    '''
    params:
    x_T is the point to track and is 1x2 and contains [x_T y_T].
    r_T is radius of patch to track
    returns:
    patch is (2*r_T+1)x(2*r_T+1) and arranged consistently with the input image I.
    '''
    patch = np.zeros((2*r_T+1, 2*r_T+1))

    for x in range(-r_T,r_T):
        for y in range(-r_T,r_T):
            pre_warp = np.asarray([x,y])
            warped = x_T + np.hstack((pre_warp,1)).dot(np.transpose(W))
            patch[y + r_T + 1, x + r_T + 1] = I[int(warped[1]), int(warped[0])]

    return patch

def track_brute_force(I_R, I, x_T, r_T, r_D):
    '''
    Params:
    I_R: reference image
    I: image to track point in
    x_T: point to track, expressed as [x y]=[col row]
    r_T: radius of patch to track
    r_D: radius of patch to search dx within
    returns:
    dx: translation that best explains where x_T is in image I
    ssds: SSDs for all values of dx within the patch defined by center x_T and radius r_D.

    '''
    dx_best = (0,0)
    ssds = np.zeros((2*r_D, 2*r_D))
    template = get_warped_patch(I_R, get_sim_warp(0,0,0,1), x_T, r_T)

    for dx in range(-r_D,r_D):
        for dy in range(-r_D,r_D):
            candidate = get_warped_patch(I, get_sim_warp(dx,dy,0,1), x_T, r_T)
            ssd = np.sum(np.sum(np.square(template - candidate)));
            ssds[dx + r_D, dy + r_D] = ssd

    index = np.unravel_index(np.argmin(ssds, axis=None), ssds.shape)
    dx_best = np.asarray([index[0], index[1]]) - r_D - 1

    return dx_best, ssds





def track_KLT_robust():
    pass

def track_KLT():
    pass

def plot_matches():
    pass
