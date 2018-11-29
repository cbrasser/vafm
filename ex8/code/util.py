import numpy as np

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
