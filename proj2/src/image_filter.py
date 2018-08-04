import cv2
import numpy as np

def get_sobel_binary(img_rgb):
    sobel_threshold = (20,100)
    sobel_ksize = 3
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,sobel_ksize)
    sobelx_mag = np.sqrt(np.square(sobelx))
    sobelx_mag_scaled = np.uint8(sobelx_mag / np.max(sobelx_mag) * 255)
    binary_sobel = np.zeros_like(img_gray)
    binary_sobel[(sobelx_mag_scaled >= sobel_threshold[0]) & (sobelx_mag_scaled <= sobel_threshold[1])] = 1

    return binary_sobel

def get_s_channel_binary(img_rgb):
    s_threshold = (170,255)
    img_hls = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HLS)
    s_channel= img_hls[:,:,-1]
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1
    
    return binary_s