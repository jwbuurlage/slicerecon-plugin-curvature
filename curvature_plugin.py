import numpy as np
import slicerecon

import scipy.io
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from skimage.filters import threshold_otsu
from matplotlib import cm
import time

# TODO import Hans' code
# TODO Compute a numerical value for curvature
# TODO Modify image with "curvature at boundary"


def CurvatureBoundary(binary_slice, downsamplefactor=4):
    center = np.array(ndimage.measurements.center_of_mass(binary_slice))
    contours = measure.find_contours(binary_slice, 0.5)
    contour = contours[0]  #select biggest contour
    contour_down = contour[::downsamplefactor]

    # Determine circumference to calculate curvature, NaNs are removed by interpolation
    X = np.zeros((contour_down.shape[0], contour_down.shape[1] + 1))
    X_period = np.zeros((contour_down.shape[0] + 2, contour_down.shape[1] + 1))

    X[:, :-1] = contour_down
    X_period[1:X_period.shape[0] - 1, :] = X  # determine periodic boundaries
    X_period[0, :] = X[X.shape[0] - 1]
    X_period[X_period.shape[0] - 1, :] = X[0, :]

    N = X.shape[0]
    R = np.zeros((N, 1))
    R.fill(np.nan)
    K = np.zeros((N, 3))

    for i in range(0, N):
        A = X_period[i + 1].T
        B = X_period[i].T
        C = X_period[i + 2].T
        D = np.cross(B - A, C - A)

        ac = np.linalg.norm(A - C)
        ab = np.linalg.norm(A - B)

        E = np.cross(D, B - A)
        F = np.cross(D, C - A)
        G = (ac**2 * E - ab**2 * F) / np.linalg.norm(D)**2 / 2
        R[i] = np.linalg.norm(G)

        if R[i] == 0:
            K[i, :] = G.T
        else:
            K[i, :] = G.T / R[i]**2

    ok = ~np.isnan(R)
    xp = ok.ravel().nonzero()[0]
    fp = R[~np.isnan(R)]
    x_R = np.isnan(R).ravel().nonzero()[0]
    R[np.isnan(R)] = np.interp(x_R, xp, fp)
    K[np.isnan(K)] = 0

    #% Interpolate values back to a finer grid
    reggrid = np.round(np.linspace(0, X.shape[0], X.shape[0], endpoint=True))
    smallgrid = np.linspace(0,
                            X.shape[0],
                            X.shape[0] * downsamplefactor,
                            endpoint=True)

    x_smallgrid = np.interp(smallgrid, reggrid, X[:, 0])
    y_smallgrid = np.interp(smallgrid, reggrid, X[:, 1])
    R_smallgrid = np.interp(smallgrid, reggrid, R[:, 0])
    Kx_smallgrid = np.interp(smallgrid, reggrid, K[:, 0])
    Ky_smallgrid = np.interp(smallgrid, reggrid, K[:, 1])

    #% determine inwards or outwards to give a sign to the final curvature measure
    direction_inwards = np.array(
        [center[0] - x_smallgrid, center[1] - y_smallgrid]).T
    K_combined = np.array([Kx_smallgrid, Ky_smallgrid]).T

    theta_vec = np.rad2deg(
        np.arccos(
            np.divide(
                np.sum(np.multiply(direction_inwards, K_combined), axis=1),
                np.multiply(
                    np.sqrt(
                        np.sum(np.multiply(direction_inwards,
                                           direction_inwards),
                               axis=1)),
                    np.sqrt(np.sum(np.multiply(K_combined, K_combined),
                                   axis=1))))))
    theta_vec[np.isnan(theta_vec)] = 0
    sign = np.subtract(1 * (theta_vec < 90), 1 * (theta_vec >= 90))

    curvature = np.divide(1, np.multiply(R_smallgrid, sign))

    output = dict()
    output['xmesh'] = x_smallgrid
    output['ymesh'] = y_smallgrid
    output['R'] = R_smallgrid
    output['K'] = K_combined
    output['curvature'] = curvature

    return output


def callback(shape, xs, idx):
    xs = np.array(xs).reshape(shape)

    # blurred_slice = ndimage.median_filter(test_slice, 2)
    # global_thresh = threshold_otsu(blurred_slice)
    # binary_slice = blurred_slice > global_thresh #0.3
    # output = CurvatureBoundary(binary_slice,6)
    # curvature =   np.array(output['curvature'])
    # ymesh =   np.array(output['ymesh'])
    # xmesh =   np.array(output['xmesh'])
    # maxcurv = np.amax(curvature)
    # mincurv = np.amin(curvature)
    # send maxcurv/mincurv as 'tracker' packets

    return [shape, xs.ravel().tolist()]


p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
p.set_slice_callback(callback)

p.listen()
