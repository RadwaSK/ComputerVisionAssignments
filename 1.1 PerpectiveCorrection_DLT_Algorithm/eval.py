"""
Assignment 1 : Image Stitching and Perspective Correction
Part I: DLT Algorithm
"""
import os
import cv2 as cv
import numpy as np
import argparse

def perspective_correction(image, homography):
    new_image = np.zeros(image.shape)
    H, W, _ = image.shape
    
    new_image = cv.warpPerspective(image, homography, (W,H), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return new_image


def calculate_target_points(src_points):
    w1 = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + ((src_points[2][1] - src_points[3][1]) ** 2))
    w2 = np.sqrt(((src_points[0][0] - src_points[1][0]) ** 2) + ((src_points[0][1] - src_points[1][1]) ** 2))
    w = max(int(w1), int(w2))

    h1 = np.sqrt(((src_points[0][0] - src_points[3][0]) ** 2) + ((src_points[0][1] - src_points[3][1]) ** 2))
    h2 = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + ((src_points[2][1] - src_points[1][1]) ** 2))
    h = max(int(h1), int(h2))

    target_points = np.array([[0,0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float64)
    return target_points
    

def normalize(img_points, mean):
    img_points_homogenous = np.concatenate((img_points.T, np.ones((1, len(img_points)))), dtype=np.float64)
    
    T = np.array([[1/np.sqrt(2), 0, mean[0]],
                    [0, 1/np.sqrt(2), mean[1]],
                    [0,   0, 1]], dtype=np.float64)

    T_inv = np.linalg.pinv(T)

    img_points_nomralized = np.dot(T_inv, img_points_homogenous)[0:2].T

    return img_points_nomralized, T_inv


def denormalize(H, T1, T2):
    T2_inv = np.linalg.pinv(T2)
    homography = np.dot(np.dot(T2_inv, H), T1)
    homography /= homography[2,2]
    return homography


def dlt(src_points, target_points):
    n = len(src_points)

    A_matrix = lambda x, y, xx, yy: np.array([[-x, -y, -1, 0, 0, 0, xx*x, xx*y, xx], [0, 0, 0, -x, -y, -1, yy*x, yy*y, yy]], dtype=np.float64)
    homography_vector = np.ones((9, 1), dtype=np.float64)
    
    assert A_matrix(1, 2, 2, 3).shape[1] == homography_vector.shape[0]
    
    A_8x9 = np.zeros((2*n,9), dtype=np.float64)

    for i in range(n):
        x, y = src_points[i]
        xx, yy = target_points[i]
        A_8x9[2*i:2*i+2] = A_matrix(x, y, xx, yy)

    u, s, vt = np.linalg.svd(A_8x9)

    homography_vector = vt.T[:,-1]
    homography_vector /= homography_vector[-1]
    return homography_vector.reshape((3, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--srcdir", help = "path to the images directory", default='images')
    ap.add_argument("-n", "--gtdir", help = "path to npy groundtruth directory", default='gt')
    ap.add_argument("--norm", action="store_true", default=False)
    ap.add_argument("-s", "--savePath", help='Path to directory to save output images', default='results')
    args = ap.parse_args()

    out_path = args.savePath
    total_error = 0
    files = os.listdir(args.srcdir)
    for i, image_path in enumerate(files):
        image = cv.imread(os.path.join(args.srcdir, image_path))
        H, W, _ = image.shape 
        npy_file = os.path.join(args.gtdir, image_path.split('/')[-1].replace('png', 'npy'))

        gt = np.load(npy_file, allow_pickle=True).item()

        src_points = gt['points']
        homography_gt = gt['homography']

        # Calculate target points using source points
        target_points = calculate_target_points(src_points)

        # In case of not normalizing, and using DLT algorithm right away [as in Zisserman & Hartley - Multiple View Geometry in Computer Vision book
        # Algorithm 4.1. Published by Cambridge University Press (2003)]
        if not args.norm:
            homography = dlt(src_points, target_points)
        
        # To use normalized DLT algorithm, [as in Zisserman & Hartley - Multiple View Geometry in Computer Vision book
        # Algorithm 4.2. Published by Cambridge University Press (2003)]
        else:
            # Get mean of X range and Y range
            mean = [np.mean(range(W)), np.mean(range(H))]

            # Normalize source points
            normalized_src_points, T = normalize(src_points, mean)
            
            # Normalize target points
            normalized_target_points, T2 = normalize(target_points, mean)
            
            # Calculated Homography matrix using DLT algorithm using normalized set of points
            H = dlt(normalized_src_points, normalized_target_points)
            
            # Denormalize the homography matrix
            homography = denormalize(H, T, T2)
            
        # Apply perspective correction on image, once using the calculated homography & obtaining the corrected image, 
        # cropped to include only the license plate
        correct_img = perspective_correction(image, homography_gt)[0:int(target_points[2,1]), 0:int(target_points[2,0])]
        predicted_img = perspective_correction(image, homography)[0:int(target_points[2,1]), 0:int(target_points[2,0])]
        
        # Calculating the MSE between homography & homography ground truth
        error = np.linalg.norm(homography.flatten() - homography_gt.flatten())**2
        total_error += error

        # Saving resulting images in path given in args.savePath (out_path)
        cv.imwrite(os.path.join(out_path, str(i) + '. corrected_cropped_gt_', image_path), correct_img)
        cv.imwrite(os.path.join(out_path, str(i) + '. corrected_cropped_', image_path), predicted_img)
        cv.imwrite(os.path.join(out_path, str(i) + '. orig_', image_path), image)
        
        print("MSE at #%d is: %lf" % (i, error))

    print('Total Mean Squared Error: ', total_error/len(files))


if __name__ == "__main__":
    main()
