import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        height -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)


    points = points.reshape(4, 3).T      # (3, 4)
    H = np.vstack((Rt, np.array([0, 0, 0, 1])))
    cal_points = np.linalg.inv(K) @ points     # (3, 4)
    cal_points /= cal_points[-1, :]
    cal_points *= depth
    cal_points = np.vstack((cal_points, np.array([1, 1, 1, 1])))   # (4, 4)
    world_points = np.linalg.inv(H) @ cal_points
    points = world_points[:-1, :].T.reshape(2, 2, 3)
 
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """

    h = points.shape[0]
    w = points.shape[1]
    points = points.reshape(h * w, 3)
    points = np.hstack((points, np.ones(h * w).reshape(-1, 1)))
    proj_points = (K @ Rt @ points.T).T
    proj_points /= (proj_points[:, -1].reshape(-1, 1))
    points = proj_points[:, :-1].reshape(h, w, 2)

    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    ref_points = np.array([[0, 0],
                           [width, 0],
                           [0, height],
                           [width, height]])

    first_proj = backproject_fn(K_ref, width, height, depth, Rt_ref)
    second_proj = project_fn(K_neighbor, Rt_neighbor, first_proj)

    h1, w1 = second_proj.shape[:-1]
    second_proj = second_proj.reshape((h1 * w1, 2))
    h, _ = cv2.findHomography(second_proj, ref_points)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, h, dsize=(width, height))
    

    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    src_mean = np.mean(src, axis = 2)
    dst_mean = np.mean(dst, axis = 2)
    src_mean = src_mean[:, :, None, :]
    dst_mean = dst_mean[:, :, None, :]
    zncc = np.sum((src - src_mean) * (dst - dst_mean), axis=2) / (np.std(src, axis=2) * np.std(dst, axis=2) + EPS)
    zncc = np.sum(zncc, axis=2)
  

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))


    xyz_cam = np.zeros((dep_map.shape[0], dep_map.shape[1], 3))
    xyz_cam[:, :, 0] = (_u - K[0, 2]) * dep_map / K[0, 0]
    xyz_cam[:, :, 1] = (_v - K[1, 2]) * dep_map / K[1, 1]
    xyz_cam[:, :, 2] = dep_map
    

    return xyz_cam

