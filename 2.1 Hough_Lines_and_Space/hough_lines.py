import os
import numpy as np
import cv2 as cv
import argparse
import matplotlib.pyplot as plt


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--srcdir", help = "path to the images directory", default='images')
	ap.add_argument("-o", "--outdir", help='path to directory to save output images in', default='results')
	args = ap.parse_args()

	in_path = args.srcdir
	images_names = os.listdir(in_path)
	out_path = args.outdir
	os.makedirs(out_path, exist_ok=True)

	for img_name in images_names:
		img_path = os.path.join(in_path, img_name)
		img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
		img_blurred = cv.GaussianBlur(img, (5, 5), 1.5)
		kernel = np.ones((5,5), np.uint8)
		canny = cv.Canny(img_blurred, 100, 170)
		# canny = cv.dilate(canny, kernel, iterations=1)
		# canny = cv.erode(canny, kernel, iterations=1)
		H, W = canny.shape
		max_distance = int(np.sqrt(H**2 + W**2))
		
		H_mat, thetas, values = hough(canny, max_distance, 1, 1)
		H_mat, peaks, max_indices = hough_peaks(H_mat, 7)

		visualize_hough_lines_space(H_mat, out_path, img_name)

		lines = hough_lines(peaks, max_distance)
		
		visualize_hough_lines(img, lines, out_path, img_name)



def visualize_hough_lines_space(H, out_path, name):
	fig = plt.figure(figsize=(10, 10))
	plt.imshow(H, origin='lower', aspect='auto')
	plt.title('Hough Space')
	plt.ylabel('R')
	plt.xlabel('Theta')
	plt.savefig(os.path.join(out_path, 'HoughSpace_' + name))
	plt.clf()


def visualize_hough_lines(img, lines, out_path, name):
	lines_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	for y1, x1, y2, x2 in lines:
		lines_image = cv.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

	cv.imwrite(os.path.join(out_path, 'HoughLines_' + name), lines_image)


def hough_lines(peaks, max_dist):
	lines = []
	for r, theta in peaks:
		lines.append(polar2cartasian(r, theta, max_dist))
	return lines


def polar2cartasian(r, theta, max_dist):
    cx = int(r * np.cos(theta))
    cy = int(r * np.sin(theta))
    min_val = min(cx, cy)
    if min_val < 0:
    	cx += min_val
    	cy += min_val

    x2 = int(cx + max_dist * np.sin(theta))
    y2 = int(cy + max_dist * np.cos(theta))
    return cx, cy, x2, y2


def hough(img, max_r, angle_range=1, ro_range=1):
	edge_points = np.where(img != 0)
	edge_points = np.vstack((edge_points[1], edge_points[0])).T

	angles = np.arange(-90, 90, angle_range)
	ros = np.arange(-max_r, max_r + 1, ro_range)
	H = np.zeros((len(ros), len(angles)))
	
	for y, x in edge_points:
		for angle in angles:
			rad_angle = np.deg2rad(angle)
			r = int(x * np.cos(rad_angle) + y * np.sin(rad_angle))
			## to store it in index indicating r, it has to be +ve
			## so we add max_r so that it is always >= 0
			# r_indx = r + max_r
			angle_indx = angle + 90
			H[r, angle_indx] += 1
	return H, angles, ros


def hough_peaks(H, peaks_num=5):
	max_indices = np.argpartition(np.ravel(H), -peaks_num)[-peaks_num:]
	max_indices = np.column_stack(np.unravel_index(max_indices, H.shape))
	peaks = []
	for r, theta in max_indices:
		H[r, theta] = 255
		theta = np.deg2rad(theta)
		peaks.append([r, theta])

	return H, peaks, max_indices



if __name__ == '__main__':
	main()