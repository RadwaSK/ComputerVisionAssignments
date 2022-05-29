import os
import argparse
import numpy as np
import cv2 as cv

global center
global rad
center, rad = None, None

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
		res_vid_name1 = os.path.join(out_path, img_name.split('.')[0] + '_contours.mp4')
		res_path1 = os.path.join(out_path, img_name)
		
		res_vid_name2 = os.path.join(out_path, img_name.split('.')[0] + '_contours_Dynamic.mp4')
		res_path2 = os.path.join(out_path, img_name.split('.')[0] + '_Dynamic.jpg')

		img_name = os.path.join(in_path, img_name)
		img = cv.imread(img_name)
		# orig = img.copy
		H, W, _ = img.shape
		
		global rad, center
		tries = 0
		while (rad is None or center is None) and tries < 5:
			print("\nPlease click using the left mouse button in the center location\nthen hold till the end of the radius & release\nAfter that you can click any keyboard key to close the window")
			cv.imshow('gray', img)
			cv.setMouseCallback("gray", capture_points)
			k = cv.waitKey()
			cv.destroyAllWindows()
			tries += 1

		if center is None:
			print('\n====================================================')
			print('\tFailed to capture center & radius')
			print('====================================================')
			continue
		
		rad = np.sqrt((rad[0] - center[0])**2 + (rad[1] - center[1])**2)
		init_contours = get_contour_points(center, rad, W, H)

		init_contours = order_points(init_contours)

		draw_cont(img, init_contours, True)

		contours = snakes(img, init_contours, res_vid_name1)

		res = draw_cont(img, contours, True)

		cv.imwrite(res_path1, res)

		rad, center = None, None



def capture_points(event, x, y, flags, param):
	global center, rad
	if event == cv.EVENT_LBUTTONDOWN:
		center = (x, y)
	elif event == cv.EVENT_LBUTTONUP:
		rad = (x, y)
		

def get_contour_points(c, rad, W, H):
	theta = 3
	st_angle = -180
	en_angle = 180
	points = []
	for angle in range(st_angle, en_angle, theta):
		rad_angle = angle / 180.0 * np.pi
		x = int(c[0] + rad  * np.cos(angle))
		if x < 0:
			x = 0
		elif x >= W:
			x = W
		y = int(c[1] + rad * np.sin(angle))
		if y < 0:
			y = 0
		elif y >= H:
			y = H
		points.append([x, y])

	return np.array(points)


def order_points(points):
	new_points = []
	points = sorted(points, key=lambda x: (x[0], -x[1]))
	upper_half = []
	lower_half = []

	limit = points[0][1]

	for p in points:
		if p[1] <= limit:
			upper_half.append(p)
		else:
			lower_half.append(p)

	upper_half = sorted(upper_half, key=lambda x: (x[0], -x[1]))
	upper_left = []
	upper_right = []

	prev = None
	right = False
	for p in upper_half:
		if not right:
			if prev is None:
				upper_left.append(p)
				prev = p
			else:
				if p[1] > prev[1]:
					upper_right.append(p)
					right = True
				else:
					upper_left.append(p)
				prev = p
		else:
			upper_right.append(p)

	upper_right = sorted(upper_right, key=lambda x: (x[0], x[1]))

	lower_half = sorted(lower_half, key=lambda x: (-x[0], x[1]))

	lower_left = []
	lower_right = []

	prev = None
	left = False
	for p in lower_half:
		if not left:
			if prev is None:
				lower_right.append(p)
				prev = p
			else:
				if p[1] < prev[1]:
					lower_left.append(p)
					left = True
				else:
					lower_right.append(p)
				prev = p
		else:
			lower_left.append(p)

	lower_left = sorted(lower_left, key=lambda x: -x[0])

	assert len(points) == (len(upper_left) + len(upper_right) + len(lower_left) + len(lower_right))

	points = np.concatenate((upper_left, upper_right, lower_right, lower_left))
	
	return points


def get_average_dist(points):
	distances = []
	prev = None
	for p in points:
		x, y = p
		if prev is None:
			prev = [x, y]
		else:
			dist = np.sqrt((x - prev[0])**2 + (y - prev[1])**2)
			distances.append(dist)
			prev = [x, y]

	return np.average(distances)


def draw_cont(orig_img, points, draw=False):
	img = orig_img.copy()
	global center
	img = cv.circle(img, center, color=(255,0,0), radius=3, thickness=3)
	prev = None
	for p in points:
		img = cv.circle(img, p, color=(0,0,255), radius=3, thickness=5)
		if prev is None:
			prev = p
		else:
			img = cv.line(img, p, prev, color=(0,0,255), thickness=3)
			prev = p
	if draw:
		cv.imshow("contours", img)
		cv.waitKey()
		cv.destroyAllWindows()
	return img


def snakes(img, points, vid_name, alpha=0.01, beta=0.4, iterations=100, dynamic=False, enhance_initially=False):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	eroded = cv.erode(gray, np.ones((3,3)), iterations = 5)
	H, W = gray.shape
	kernel = np.ones((5,5)) / 25
	# blurred = cv.filter2D(gray, -1, kernel)
	_, binary = cv.threshold(eroded, 100, 255, cv.THRESH_BINARY)
	if enhance_initially:
		points = enhance_init_contours(binary, points)
		points = order_points(points)
		draw_cont(img, points, True)
	edge = cv.Canny(gray, 100, 200)
	edge = cv.filter2D(edge, -1, kernel)
	
	frames = []
	video = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'XVID'), 30, frameSize=(W,H))
	
	for i in range(iterations):
		updated = False
		d = get_average_dist(points)
		for j in range(0, len(points)):
			x, y = points[j]
			
			if j == 0:
				prevp = points[-1]
			else:
				prevp = points[j-1]
			if j == len(points) - 1:
				nextp = points[0]
			else:
				nextp = points[j+1]

			bestX, bestY = x, y
			minE = calc_energy(gray, edge[y,x], alpha, beta, prevp, (x, y), nextp, d)
			changed = False
			no_solution = False
			window_size = 30

			if dynamic:
				while window_size < 150 and (not updated):
					for yy in range(y - window_size, y + window_size + 1):
						for xx in range(x - window_size,x + window_size + 1):
							E = calc_energy(gray, edge[yy, xx], alpha, beta, prevp, (xx,yy), nextp, d)
							if E < minE:
								updated = True
								minE = E
								bestX, bestY = xx, yy
					window_size += 5
			else:
				for yy in range(y - window_size, y + window_size + 1):
					for xx in range(x - window_size,x + window_size + 1):
						E = calc_energy(gray, edge[yy, xx], alpha, beta, prevp, (xx,yy), nextp, d)
						if E < minE:
							updated = True
							minE = E
							bestX, bestY = xx, yy
				
			points[j] = (bestX, bestY)
			ret = draw_cont(img, points)
			if ret is not None:
				frames.append(ret)
			else:
				print('Error!')
		if not updated:
			break
	print("Number of frames =", len(frames))
	for frame in frames:
		video.write(frame)

	video = video.release()
	cv.destroyAllWindows()
		
	return points


def calc_energy(img, edge_img_p, alpha, beta, prevp, point, nextp, avg_dist):
	x, y = prevp
	X, Y = point
	xx, yy = nextp
	E_int = alpha * (avg_dist - np.abs(img[yy,xx] - img[Y,X]))**2 + beta * (img[yy,xx] - 2*img[Y,X] + img[y,x])**2
	E_ext = -edge_img_p**2
	return E_int + E_ext


def enhance_init_contours(img, points):
	window_size = 100
	for k, p in enumerate(points):
		# if k > 0 and k < len(points):
		# 	p0 = points[k-1]
		# 	p1 = points[k+1]
		if img[p[1],p[0]] == 255:
			done = False
			for j in range(p[1] - window_size, p[1] + window_size + 1):
				for i in range(p[0] - window_size, p[0] + window_size + 1):
					if img[j, i] != 255:
						points[k] = (i, j)
						done = True
						break
				if done:
					break
	return points

if __name__ == '__main__':
	main()