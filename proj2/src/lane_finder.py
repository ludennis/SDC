import src.camera_calibrator as cc
import src.image_filter as imgf
import numpy as np
import cv2


def get_binary_warped(img_rgb):
    # parameters for undistortion
    camera_img_dir = 'camera_cal/'
    nx,ny = 9,6
    objpoints,imgpoints = cc.get_obj_img_points(camera_img_dir,nx,ny)
    
    # parameters for transformation
    bl_src, tl_src, tr_src, br_src = (252,690), (552,480), (735,480), (1070,690)
    bl_dst, tl_dst, tr_dst, br_dst = (300,690), (300,10), (900,10), (900,690)
    src = np.float32([tl_src, tr_src, bl_src, br_src])
    dst = np.float32([tl_dst, tr_dst, bl_dst, br_dst])
    
    # get undistorted img
    img_undistorted = cc.undistort_img(img_rgb,objpoints,imgpoints)
    
    # get binary combination of sobel x and s-channel
    binary_sobel = imgf.get_sobel_binary(img_undistorted)
    binary_s = imgf.get_s_channel_binary(img_undistorted)
    binary_combined = np.zeros_like(binary_s)
    binary_combined[(binary_sobel==1) | (binary_s==1)] = 1
        
    binary_combined_warped, trans_M = cc.transform_perspective(binary_combined,src,dst)
    
    return binary_combined_warped, trans_M
    

def get_lane_pixels(img_binary_warped):
    # img size
    size_x = img_binary_warped.shape[1]
    size_y = img_binary_warped.shape[0]
            
    # output image for visualization
    out_img = np.dstack((img_binary_warped*255,img_binary_warped*255,img_binary_warped*255))
        
    # sliding window parameters
    n_windows = 9
    min_pixels = 50
    window_width = 100
    window_height = np.int(size_y//n_windows)
    
    # get initial x point of sliding window by using histograms
    histogram = np.sum(img_binary_warped[size_y//2:,:],axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    left_x = np.argmax(histogram[:midpoint])
    right_x = np.argmax(histogram[midpoint:]) + midpoint
    
    # get position of where x and y pixels 
    nonzero = img_binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # list of pixels positions (indices) to keep
    left_indices = []
    right_indices = []
    
    left_x_cur = left_x
    right_x_cur = right_x
    
    # get sliding windows
    for window in range(n_windows):
        
        left_low_x, left_high_x = left_x_cur-window_width, left_x_cur+window_width
        right_low_x, right_high_x = right_x_cur-window_width, right_x_cur+window_width
        low_y, high_y = size_y-(window+1)*window_height, size_y-(window)*window_height
        
        # draw rectangles for both left and right lane
        cv2.rectangle(out_img, (left_low_x,low_y),(left_high_x,high_y),(0,255,0),2)
        
        cv2.rectangle(out_img, (right_low_x,low_y),(right_high_x,high_y),(0,255,0),2)
                
        # include pixels within rectangle
        left_indices_in_window = ((nonzeroy >= low_y) & (nonzeroy < high_y) & 
                                 (nonzerox >= left_low_x) & (nonzerox < left_high_x)).nonzero()[0]
        right_indices_in_window = ((nonzeroy >= low_y) & (nonzeroy < high_y) &
                                  (nonzerox >= right_low_x) & (nonzerox < right_high_x)).nonzero()[0]
                
        left_indices.append(left_indices_in_window)
        right_indices.append(right_indices_in_window)
        
        
        # shift the midpoints if the sliding window has more than min_pixels
        if len(left_indices_in_window) > min_pixels:
            left_x_cur = np.int(np.mean(nonzerox[left_indices_in_window]))
        if len(right_indices_in_window) > min_pixels:
            right_x_cur = np.int(np.mean(nonzerox[right_indices_in_window]))
    
    # put all pixels positions together into one list
    left_indices = np.concatenate(left_indices)
    right_indices = np.concatenate(right_indices)
    
    # extract all the pixels for returning
    left_x_pixels = nonzerox[left_indices]
    left_y_pixels = nonzeroy[left_indices]
    right_x_pixels = nonzerox[right_indices]
    right_y_pixels = nonzeroy[right_indices]
    
    return left_x_pixels, left_y_pixels, right_x_pixels, right_y_pixels, out_img


'''
    takes in a warped-binary image
    outputs polynomial fits for both left and right lane and the warped image with rectangles drawn
'''
def get_poly_lines(img_binary_warped):
    
    # get pixels bounded by all the sliding windows
    left_x_pixels, left_y_pixels, right_x_pixels, right_y_pixels, out_img = get_lane_pixels(img_binary_warped)
    
    # use all the pixels gathered from sliding windows to fit polynomial lines
    left_fit = np.polyfit(left_y_pixels, left_x_pixels,2)
    right_fit = np.polyfit(right_y_pixels, right_x_pixels,2)
    
    # get x's and y's to plot
    y = np.linspace(0, img_binary_warped.shape[0]-1, img_binary_warped.shape[0])
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    
    # show pixels captured by windows
    out_img[left_y_pixels, left_x_pixels] = [255,0,0]
    out_img[right_y_pixels, right_x_pixels] = [0, 0, 255]
    
    return left_fit, right_fit, left_x, right_x, y,out_img


'''
    Input: binary warped image
    Output: left curvature radius, right curvature radius (both in meters)
    Finds the curvature radius in meters. Radius values should be around 1000 meters
'''
def measure_curvature_real(img_binary_warped):
    
    left_x_pixels, left_y_pixels, right_x_pixels, right_y_pixels, out_img = get_lane_pixels(img_binary_warped)

    y_eval = img_binary_warped.shape[0]
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(left_y_pixels*ym_per_pix, left_x_pixels*xm_per_pix,2)
    right_fit_cr = np.polyfit(right_y_pixels*ym_per_pix, right_x_pixels*xm_per_pix,2)
    
    left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5 / np.abs(2*left_fit_cr[0])  
    right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5 / np.abs(2*right_fit_cr[0])

    print ("left curvature radiance: {} / right curvature radiance: {}".format(left_curverad, right_curverad))
    
    return left_curverad, right_curverad


'''
    Checks the following:
        1. Similiar radius for both curves
        2. Separated approximately the right distance apart
        3. Roughly parallel to each other
        
    Returns a boolean
'''
def sanity_check(left_curverad, right_curverad):
    pass


def draw_lane_lines_polygon(img_rgb):
    
    # get binary_warped img and left, right fits
    img_binary_warped, trans_M = get_binary_warped(img_rgb)
    left_fit, right_fit, left_x, right_x, y, out_img = get_poly_lines(img_binary_warped)
    
    # create a blank image in warped space
    warped_zero = np.zeros_like(img_binary_warped).astype(np.uint8)
    warped_color = np.dstack((warped_zero, warped_zero, warped_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_x, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    pts = np.hstack((pts_left, pts_right))
    
    # draws the polygon in warped image
    cv2.fillPoly(warped_color, np.int_(pts), (0,255,0))
    
    # project polygon to original image space
    inv_M = np.linalg.inv(trans_M)
    polygon_unwarped = cv2.warpPerspective(warped_color, inv_M, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # add polygon to original image
    img_rgb_polygon = cv2.addWeighted(img_rgb, 1, polygon_unwarped, 0.3, 0)

    return img_rgb_polygon
