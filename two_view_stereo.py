import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding


    H_i = K_i_corr @ R_irect @ np.linalg.inv(K_i)
    H_i /= H_i[-1, -1]
    H_j = K_j_corr @ R_jrect @ np.linalg.inv(K_j)
    H_j /= H_j[-1, -1]

    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    R_ji = R_wi @ R_wj.T
    T_ji = T_wi - R_ji @ T_wj
    B = np.linalg.norm(R_ji @ np.array([[0], [0], [0]]) + T_ji)

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)

    r2 = e_i / np.linalg.norm(e_i)
    r1 = np.cross(r2, np.array([0, 0, 1])) / np.linalg.norm(np.cross(r2, np.array([0, 0, 1])))
    r3 = np.cross(r1, r2) / np.linalg.norm(np.cross(r1, r2))

    R_irect = np.vstack((r1, r2, r3))

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]


    # M = src.shape[0]
    # N = dst.shape[0]
    # src_ch_1 = src[:, :, 0]       # (475, 25)
    # src_ch_2 = src[:, :, 1]       # (475, 25)
    # src_ch_3 = src[:, :, 2]       # (475, 25)
    #
    # dst_ch_1 = dst[:, :, 0]       # (475, 25)
    # dst_ch_2 = dst[:, :, 1]       # (475, 25)
    # dst_ch_3 = dst[:, :, 2]       # (475, 25)
    #
    # cr_ch_1 = np.zeros((475, 475))
    # cr_ch_2 = np.zeros((475, 475))
    # cr_ch_3 = np.zeros((475, 475))
    # for i in range(M):
    #     for j in range(N):
    #         cr_ch_1[i, j] = np.sum(np.square(src_ch_1[i] - dst_ch_1[j]))
    #         cr_ch_2[i, j] = np.sum(np.square(src_ch_2[i] - dst_ch_2[j]))
    #         cr_ch_3[i, j] = np.sum(np.square(src_ch_3[i] - dst_ch_3[j]))
    #
    # ssd = cr_ch_1 + cr_ch_2 + cr_ch_3

    src = src[: , np.newaxis, :, :]
    dst = dst[np.newaxis, :, : :]
    # ssd = src - dst

    ssd_channel_1 = np.sum((src[:,:,:,0]-dst[:,:,:,0])**2, axis = 2)
    ssd_channel_2 = np.sum((src[:,:,:,1]-dst[:,:,:,1])**2, axis = 2)
    ssd_channel_3 = np.sum((src[:,:,:,2]-dst[:,:,:,2])**2, axis = 2)
    # print(ssd_channel_1.shape)

    ssd = ssd_channel_1 + ssd_channel_2 + ssd_channel_3

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    # M = src.shape[0]
    # N = dst.shape[0]
    # src_ch_1 = src[:, :, 0]  # (475, 25)
    # src_ch_2 = src[:, :, 1]  # (475, 25)
    # src_ch_3 = src[:, :, 2]  # (475, 25)
    #
    # dst_ch_1 = dst[:, :, 0]  # (475, 25)
    # dst_ch_2 = dst[:, :, 1]  # (475, 25)
    # dst_ch_3 = dst[:, :, 2]  # (475, 25)
    #
    # cr_ch_1 = np.zeros((475, 475))
    # cr_ch_2 = np.zeros((475, 475))
    # cr_ch_3 = np.zeros((475, 475))
    # for i in range(M):
    #     for j in range(N):
    #         cr_ch_1[i, j] = np.sum(np.abs(src_ch_1[i] - dst_ch_1[j]))
    #         cr_ch_2[i, j] = np.sum(np.abs(src_ch_2[i] - dst_ch_2[j]))
    #         cr_ch_3[i, j] = np.sum(np.abs(src_ch_3[i] - dst_ch_3[j]))
    #
    # sad = cr_ch_1 + cr_ch_2 + cr_ch_3

    src = src[: , np.newaxis, :, :]
    dst = dst[np.newaxis, :, : :]

    sad_channel_1 = np.sum(np.abs(src[:,:,:,0]-dst[:,:,:,0]), axis = 2)
    sad_channel_2 = np.sum(np.abs(src[:,:,:,1]-dst[:,:,:,1]), axis = 2)
    sad_channel_3 = np.sum(np.abs(src[:,:,:,2]-dst[:,:,:,2]), axis = 2)

    sad = sad_channel_1 + sad_channel_2 + sad_channel_3

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

 
    # M = src.shape[0]
    # N = dst.shape[0]
    # src_ch_1 = src[:, :, 0]  # (475, 25)
    # src_ch_2 = src[:, :, 1]  # (475, 25)
    # src_ch_3 = src[:, :, 2]  # (475, 25)
    #
    # dst_ch_1 = dst[:, :, 0]  # (475, 25)
    # dst_ch_2 = dst[:, :, 1]  # (475, 25)
    # dst_ch_3 = dst[:, :, 2]  # (475, 25)
    #
    # cr_ch_1 = np.zeros((475, 475))
    # cr_ch_2 = np.zeros((475, 475))
    # cr_ch_3 = np.zeros((475, 475))
    #
    # for i in range(M):
    #     for j in range(N):
    #         w_i_bar_ch_1 = np.mean(src_ch_1[i])
    #         w_j_bar_ch_1 = np.mean(dst_ch_1[j])
    #         sigma_w_i_ch_1 = np.std(src_ch_1[i])
    #         sigma_w_j_ch_1 = np.std(dst_ch_1[j])
    #
    #         cr_ch_1[i, j] = np.sum((src_ch_1[i] - w_i_bar_ch_1)*(dst_ch_1[j] - w_j_bar_ch_1)) / ((sigma_w_i_ch_1*sigma_w_j_ch_1) + EPS)
    #
    #         w_i_bar_ch_2 = np.mean(src_ch_2[i])
    #         w_j_bar_ch_2 = np.mean(dst_ch_2[j])
    #         sigma_w_i_ch_2 = np.std(src_ch_2[i])
    #         sigma_w_j_ch_2 = np.std(dst_ch_2[j])
    #
    #         cr_ch_2[i, j] = np.sum((src_ch_2[i] - w_i_bar_ch_2)*(dst_ch_2[j] - w_j_bar_ch_2)) / ((sigma_w_i_ch_2*sigma_w_j_ch_2) + EPS)
    #
    #         w_i_bar_ch_3 = np.mean(src_ch_3[i])
    #         w_j_bar_ch_3 = np.mean(dst_ch_3[j])
    #         sigma_w_i_ch_3 = np.std(src_ch_3[i])
    #         sigma_w_j_ch_3 = np.std(dst_ch_3[j])
    #
    #         cr_ch_3[i, j] = np.sum((src_ch_3[i] - w_i_bar_ch_3)*(dst_ch_3[j] - w_j_bar_ch_3)) / ((sigma_w_i_ch_3*sigma_w_j_ch_3) + EPS)
    #
    # zncc = cr_ch_1 + cr_ch_2 + cr_ch_3


    ch_1_src = src[:,:,0]
    ch_2_src = src[:,:,1]
    ch_3_src = src[:,:,2]

    ch_1_dst = dst[:,:,0]
    ch_2_dst = dst[:,:,1]
    ch_3_dst = dst[:,:,2]

    ch_1_mean_src = np.mean(ch_1_src, axis = 1).reshape((-1, 1))
    ch_2_mean_src = np.mean(ch_2_src, axis = 1).reshape((-1, 1))
    ch_3_mean_src = np.mean(ch_3_src, axis = 1).reshape((-1, 1))

    ch_1_mean_dst = np.mean(ch_1_dst, axis = 1).reshape((-1, 1))
    ch_2_mean_dst = np.mean(ch_2_dst, axis = 1).reshape((-1, 1))
    ch_3_mean_dst = np.mean(ch_3_dst, axis = 1).reshape((-1, 1))

    sig_1_src = np.std(ch_1_src, axis = 1).reshape((-1, 1))
    sig_2_src = np.std(ch_2_src, axis = 1).reshape((-1, 1))
    sig_3_src = np.std(ch_3_src, axis = 1).reshape((-1, 1))

    sig_1_dst = np.std(ch_1_dst, axis = 1).reshape((-1, 1))
    sig_2_dst = np.std(ch_2_dst, axis = 1).reshape((-1, 1))
    sig_3_dst = np.std(ch_3_dst, axis = 1).reshape((-1, 1))
    a_1 = (ch_1_src - ch_1_mean_src)
    a_1 = a_1[:, np.newaxis, :]

    b_1 = (ch_1_dst - ch_1_mean_dst)
    b_1 = b_1[np.newaxis, :, :]

    zncc_channel_1 = np.sum((a_1 * b_1), axis = 2)/ ((sig_1_src @ sig_1_dst.T) + EPS)

    a_2 = (ch_2_src - ch_2_mean_src)
    a_2 = a_2[:, np.newaxis, :]

    b_2 = (ch_2_dst - ch_2_mean_dst)
    b_2 = b_2[np.newaxis, :, :]

    zncc_channel_2 = np.sum((a_2 * b_2), axis = 2)/ ((sig_2_src @ sig_2_dst.T) + EPS)

    a_3 = (ch_3_src - ch_3_mean_src)
    a_3 = a_3[:, np.newaxis, :]

    b_3 = (ch_3_dst - ch_3_mean_dst)
    b_3 = b_3[np.newaxis, :, :]

    zncc_channel_3 = np.sum((a_3 * b_3), axis =2)/ ((sig_3_src @ sig_3_dst.T) + EPS)

    # zncc = np.sum((zncc_channel_1, zncc_channel_2, zncc_channel_3))

    zncc = zncc_channel_1 + zncc_channel_2 + zncc_channel_3

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """


    H = image.shape[0]
    W = image.shape[1]
    padding_size = int(k_size / 2)
    image_reshaped = image.reshape(3, H, W)
    image_padded = np.pad(image_reshaped, pad_width=padding_size, mode='constant')
    if k_size != 1:
        final_image = np.zeros((H + k_size - 1, W + k_size - 1, 3))
        final_image[:, :, 0] = np.pad(image[:, :, 0], pad_width=padding_size, mode='constant')
        final_image[:, :, 1] = np.pad(image[:, :, 1], pad_width=padding_size, mode='constant')
        final_image[:, :, 2] = np.pad(image[:, :, 2], pad_width=padding_size, mode='constant')
    else:
        final_image = image_padded.reshape(H, W, 3)

    patch_buffer = np.zeros((H, W, k_size**2, 3))

    for h in range(H):
        for w in range(W):
            row = np.arange(h - padding_size, h + padding_size + 1)
            col = np.arange(w - padding_size, w + padding_size + 1)
            rows, columns = np.meshgrid(row, col)
            rows = rows + padding_size
            columns = columns + padding_size

            patch_buffer[h, w, :, 0] = final_image[rows, columns, 0].flatten()
            patch_buffer[h, w, :, 1] = final_image[rows, columns, 1].flatten()
            patch_buffer[h, w, :, 2] = final_image[rows, columns, 2].flatten()

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    h, w = rgb_i.shape[:2]

    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]

    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0

    disp_map = np.zeros((h, w), dtype = np.float64)
    lr_consistency_mask = np.zeros((h, w), dtype = np.float64)

    for u in range(w):
        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = kernel_func(buf_i, buf_j)
        best_matched_right_pixel = value.argmin(axis=1)
        best_matched_left_pixel = value[:, best_matched_right_pixel].argmin(axis=0)
        disp_map[:, u] = disp_candidates[np.arange(h), best_matched_right_pixel]
        lr_consistency_mask[:, u] = best_matched_left_pixel == vi_idx

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """


    dep_map = (K[1, 1] * B) / disp_map

    xyz_cam = np.zeros((disp_map.shape[0], disp_map.shape[1], 3))

    u, v = np.meshgrid(np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]))

    xyz_cam[:, :, 0] = (u - K[0, 2]) * dep_map / K[0, 0]
    xyz_cam[:, :, 1] = (v - K[1, 2]) * dep_map / K[1, 1]
    xyz_cam[:, :, 2] = dep_map

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    pcl_world = (R_wc.T @ pcl_cam.T - R_wc.T @ T_wc).T


    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
