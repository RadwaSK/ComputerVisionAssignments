"""
Assignment 1 : Image Stitching and Perspective Correction
Part II: Image Stitching
"""
import os
import cv2 as cv
import numpy as np
import argparse
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--srcdir", help = "path to the images directory", default='images')
    ap.add_argument("-s", "--savePath", help='Path to directory to save output images', default='results')
    args = ap.parse_args()
    
    out_path = args.savePath
    os.makedirs(out_path, exist_ok=True)
    src_path = args.srcdir
    
    test_cases = os.listdir(src_path)
    for case in test_cases:
        # read two equivalent images
        case_path = os.path.join(src_path, case)
        images = os.listdir(case_path)
        path1 = os.path.join(case_path, images[0])
        path2 = os.path.join(case_path, images[1])
        img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)

        # Calculate threshold for error for RANSAC
        H, W = img1.shape
        max_distance = np.sqrt(H**2 + W**2)
        ERROR_THRESHOLD = 0.005 * max_distance
        print("Error Threshold is", ERROR_THRESHOLD)

        # Get features from both images
        features1 = get_features(img1, False)
        features2 = get_features(img2, False)

        # Get matches features in both images
        src_points, dst_points = match_features(img1, img2, features1, features2, show=False)

        assert len(src_points) == len(dst_points)

        # Run RANSAC Algorithm to obtain Homography
        H = RANSAC(src_points, dst_points, ERROR_THRESHOLD, 4)

        # Obtain Homography using RANSAC OpenCV for comparison
        H_cv, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, 5)
        H_error = np.linalg.norm(H.flatten() - H_cv.flatten())**2
        print("Homography error using RANSAC:", H_error)

        # Stitching both images using different homographies
        stitched_image_1, refined_image_1 = stitch_image(img1, img2, H)
        stitched_image_11, refined_image_11 = stitch_image(img1, img2, H_cv)
        
        # Run LMeds (LMeds) to obtain Homography
        H2 = LMeds(src_points, dst_points)

        # Obtain Homography using RANSAC OpenCV for comparison
        H_cv2, _ = cv.findHomography(src_points, dst_points, cv.LMEDS, ERROR_THRESHOLD)
        H_error2 = np.linalg.norm(H2.flatten() - H_cv2.flatten())**2
        print("Homography error using LMEDS:", H_error2)

        # stitching both images using different homographies
        stitched_image_2, refined_image_2 = stitch_image(img1, img2, H2)
        stitched_image_22, refined_image_22 = stitch_image(img1, img2, H_cv2)

        # saving images
        save_path = os.path.join(out_path, case)
        os.makedirs(save_path, exist_ok=True)
        cv.imwrite(os.path.join(save_path, 'Stiteched_Image1_using_RANSAC.jpg'), stitched_image_1)
        cv.imwrite(os.path.join(save_path, 'Stitched_Image1_using_OpenCV_RANSAC.jpg') , stitched_image_11)
        cv.imwrite(os.path.join(save_path, 'Refined_Image1_using_RANSAC.jpg'), refined_image_1)
        cv.imwrite(os.path.join(save_path, 'Refined_Image1_using_OpenCV_RANSAC.jpg') , refined_image_11)
        cv.imwrite(os.path.join(save_path, 'Stitched_Image1_using_LMEDS.jpg'), stitched_image_2)
        cv.imwrite(os.path.join(save_path, 'Stitched_Image1_using_OpenCV_LMEDS.jpg'), stitched_image_22)
        cv.imwrite(os.path.join(save_path, 'Refined_Image1_using_LMEDS.jpg'), refined_image_2)
        cv.imwrite(os.path.join(save_path, 'Refined_Image1_using_OpenCV_LMEDS.jpg'), refined_image_22)
        print('\n=========================================================================\n')
        

def get_features(img, draw=False):
    detector = cv.ORB_create()
    kp, desc = detector.detectAndCompute(img, None)
    if draw:
        desc_img = cv.drawKeypoints(img, kp, img)
        cv.imshow('desc_img', desc_img)
        cv.waitKey()

    return kp, desc


def match_features(img1, img2, features1, features2, K=2, show=False):
    kp1, desc1 = features1
    kp2, desc2 = features2
    
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)
 
    matches = sorted(matches, key=lambda x: x.distance)
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        if match.distance <= 60:
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        
    # Draw the matched image
    if show:
        matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
        cv.imshow('matched_img', matched_img)
        cv.waitKey()

    return points1, points2


def dlt(src_points, target_points):
    n = len(src_points)

    A_matrix = lambda x, y, xx, yy: np.array([[-x, -y, -1, 0, 0, 0, xx*x, xx*y, xx], [0, 0, 0, -x, -y, -1, yy*x, yy*y, yy]], dtype=np.float64)
    homography_vector = np.ones((9, 1), dtype=np.float64)
    
    assert A_matrix(1, 2, 2, 3).shape[1] == homography_vector.shape[0]
    
    A_2Nx9 = np.zeros((2*n,9), dtype=np.float64)

    for i in range(n):
        x, y = src_points[i]
        xx, yy = target_points[i]
        A_2Nx9[2*i:2*i+2] = A_matrix(x, y, xx, yy)

    u, s, vt = np.linalg.svd(A_2Nx9)

    homography_vector = vt.T[:,-1]
    if homography_vector[-1] == 0:
        homography_vector[-1] = 10**-10
    homography_vector /= homography_vector[-1]

    if math.isnan(homography_vector[-1]):
        return np.zeros((3,3)), False

    return homography_vector.reshape((3, 3)), True


def warp_points(points, H):
    res_points = np.array([warp_point(p, H) for p in points])
    return res_points


def warp_point(p, H):
    x, y = p
    denomenator = H[2,0] * x + H[2,1] * y + H[2,2]

    numeratorX = H[0,0] * x + H[0,1] * y + H[0,2]
    numeratorY = H[1,0] * x + H[1,1] * y + H[1,2]

    return [int(numeratorX / denomenator), int(numeratorY / denomenator)]


def RANSAC(src_points, dst_points, error_threshold=5, sample_num=4, adaptive=False):
    MAX_ITER = 1000
    n_inliers = -1
    homography = None
    n = len(src_points)
    max_src_inliers, max_dst_inliers = [], []
    for i in range(MAX_ITER):
        indices = np.random.choice(len(src_points), sample_num)
        src_sample, dst_sample = src_points[indices], dst_points[indices]
        H, check = dlt(src_sample, dst_sample)
        if not check:
            continue
        
        res_points = warp_points(src_points, H)
        
        errors = np.array([np.linalg.norm(res_points[i] - dst_points[i]) for i in range(len(res_points))])
        
        if adaptive:
            err = np.array((errors**2))
            err = err[err != 0]
            median = np.median(err)
            sigma = 1.4826 * (1 + 5.0 / (n - sample_num)) * np.sqrt(median)
            error_threshold = np.sqrt(2.5 * sigma)
            

        inliers_src = src_points[errors < error_threshold]
        inliers_dst = dst_points[errors < error_threshold]


        if len(inliers_src) > n_inliers:
            n_inliers = len(inliers_src)
            max_src_inliers = inliers_src.copy()
            max_dst_inliers = inliers_dst.copy()
            homography = H

        ratio = len(inliers_src) / len(res_points)
        
        if ratio > 0.8 or i == MAX_ITER - 1:
            H, check = dlt(max_src_inliers, max_dst_inliers)
            if not check:
                continue
            print("Max ratio at which H is calculated: ", n_inliers / len(res_points))
            break

    print('Broken at i =', i)

    return homography
    

def LMeds(src_points, dst_points, sample_num=5):
    return RANSAC(src_points, dst_points, sample_num=sample_num, adaptive=True)


def perspective_correction(image, H2, homography):
    H, W = image.shape
    
    new_image = cv.warpPerspective(image, homography, (H+H2, W))
    
    return new_image


def stitch_image(img1, img2, H):
    refined_image = perspective_correction(img1, img2.shape[0], H)
    stitched_image = refined_image.copy()
    stitched_image[0:img2.shape[0], 0:img2.shape[1]] = img2
    return stitched_image, refined_image

if __name__ == "__main__":
    main()
