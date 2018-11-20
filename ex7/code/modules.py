import functools, time
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


# Utility decorators for debugging / performance measurements

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def debug(func):
    '''
    Prints Input and output information on the decorated function
    '''
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f'{k}={v!r}' for k,v in kwargs.items()]
        signature = ', '.join(args_repr + kwargs_repr)
        print(f'Calling {func.__name__}({signature})')
        result = func(*args, **kwargs)
        print(f'{func.__name__!r} returned {value!r}')
        return result
    return wrapper_debug

def timer(func):
    '''
    Invokes the decorated function and prints its runtime
    '''
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'Finished running {func.__name__!r} in {run_time:.4f} seconds')
        return result
    return wrapper_timer

# --------------------------------------------------------------------------------------------------

def parabola_ransac(data, max_noise):
    '''
    Fits a parabola from 3 randomly selected points. Does this for x iterations and returns the best
    fits.
    Data should be n x 2 where the rows correspond to the x and y coordinates of datapoints
    max_noise is the range which determines what is an outlier
    '''
    num_points_to_fit = 3
    best_guess_history = []
    max_num_inliers_history = []
    num_iters = int(np.log(0.01)/np.log(1-np.power((1-0.5),num_points_to_fit)))
    print(f'I will do {num_iters} Iterations')
    for j in range(num_iters):
        chosen_points_x, chosen_points_y = [], []
        for i in range(num_points_to_fit):
            tmp = np.random.randint(len(data[0]))
            chosen_points_x.append(data[0][tmp])
            chosen_points_y.append(data[1][tmp])
        fit = np.polyfit(chosen_points_x, chosen_points_y,2)
        num_inliers = 0
        for i,x in enumerate(data[0]):
            prediction = np.polyval(fit,x)
            truth = data[1][i]
            error = np.abs(np.abs(prediction) - np.abs(truth))
            if error < max_noise:
                num_inliers += 1
        if len(max_num_inliers_history) == 0 or num_inliers > np.max(max_num_inliers_history):
            max_num_inliers_history.append(num_inliers)
            best_guess_history.append(fit)
    return best_guess_history, max_num_inliers_history

def harris(img, parch_size, kappa):

    sobel_para = np.asarray([-1, 0, 1])
    sobel_orth = np.asarray([1, 2, 1])

    Ix = sp.signal.convolve2(np.transpose(sobel_orth), sobel_para, img, 'valid')
    Iy = sp.signal.convolve2(np.transpose(sobel_para), sobel_orth, img, 'valid')
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = np.multiply(Ix,Iy)

    patch = np.ones(patch_size, patch_size)
    pr = np.floor(patch_size / 2)  # patch radius
    sIxx = sp.signal.convolve2(Ixx, patch, 'valid')
    sIyy = sp.signal.convolve2(Iyy, patch, 'valid')
    sIxy = sp.signal.convolve2(Ixy, patch, 'valid')

    scores = (np.mult(sIxx,sIyy) - np.square(sIxy)) - np.mult(kappa,  np.square(sIxx + sIyy))

    scores[scores<0] = 0

    scores = np.pad(scores, np.array((1+pr, 1+pr)))

def select_keypoints_from_harrris_scores(scores, num, r):
    '''
    Selects the num best scores as keypoints and performs non-maximum
    % supression of a (2r + 1)*(2r + 1) box around the current maximum.
    '''
    keypoints = np.zeros(2, num)
    temp_scores = np.pad(scores, np.array((r, r)))
    for i in range(1,num):
        tilde, kp = np.max(temp_scores[:])
        row, col = np.ind2sub(np.shape(temp_scores), kp)
        kp = np.array(row,col)
        keypoints[:, i] = kp - r
        temp_scores[kp[1]-r:kp[1]+r, kp[2]-r:kp[2]+r] = np.zeros(2*r + 1, 2*r + 1)



def ransac_localization(query_image, database_image, keypoints, p_W_landmarks, K):
    '''
    query_keypoints should be 2 x 1000
    all_matches should be 1 x 1000 and correspond to the output from the match_descriptors()
    function from exercise 3
    best_inlier_mask should be 1 x num_matched and contain, only for the matched keypoints,
    0 if the match is an outlier and 1 otherwise.
    '''
    use_p3p = True
    tweaked_for_more = True

    # Parameters form exercise 3.
    harris_patch_size = 9
    harris_kappa = 0.08
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 5

    # Other parameters
    num_keypoints = 1000

    num_iterations = 2000
    pixel_tolerance = 10
    k = 6

    query_harris = harris(query_image, harris_patch_size, harris_kappa)
    query_keypoints = selectKeypoints(query_harris, num_keypoints, nonmaximum_supression_radius)
    query_descriptors = describeKeypoints(query_image, query_keypoints, descriptor_radius)
    database_descriptors = describeKeypoints(database_image, database_keypoints, descriptor_radius)
    all_matches = matchDescriptors(query_descriptors, database_descriptors, match_lambda)







    return None, None, None, None, None
