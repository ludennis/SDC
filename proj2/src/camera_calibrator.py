
import os
import cv2
import numpy as np


'''
    given image directory
    returns object points and image points
'''
def get_obj_img_points(img_dir, nx, ny):

    # preparation for object points and image points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    
    for i,img_path in enumerate(os.listdir(img_dir)):
        # reading the image in grayscale
        img_bgr = cv2.imread(img_dir+img_path)
        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(img_gray,(nx,ny),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img_gray,corners,(11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
    
    return objpoints, imgpoints


'''
    takes in a rgb image, object points, and image points
    outputs an undistorted image
'''
def undistort_img(img_rgb,objpoints, imgpoints):
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    
    # camera calibration with object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1],None,None)
    
    # undistort image
    img_undistorted = cv2.undistort(img_rgb,mtx,dist,None,mtx)
    
    return img_undistorted


'''
    takes in an undistorted image in rgb format, number of x corners, and number of y corners
    returns a perspective-transformed image if the transform can be computed, otherwise returns original undistorted image
'''
def transform_perspective(img_undistorted,src,dst):
    img_size = (img_undistorted.shape[1],img_undistorted.shape[0]) # img_size = (x_size,y_size)

    # get transform matrix and applies it to undistorted image
    M = cv2.getPerspectiveTransform(src,dst)
    img_warped = cv2.warpPerspective(img_undistorted,M,img_size)

    return img_warped, M


