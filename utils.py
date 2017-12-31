import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle



def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    sobel = np.abs(sobel)
    print(np.max(sobel))
    # sobely = np.abs(sobely)
    
    # scalex = np.uint8(sobelx * 255 / np.max(sobelx))
    scale = np.uint8(sobel * 255 / np.max(sobel))
    binary_output = np.zeros_like(scale)
    binary_output[(scale > thresh_min) & (scale < thresh_max)] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    m = np.sqrt(np.square(x) + np.square(y))
    # print(m)
    m = np.uint8(m * 255 / np.max(m))
    binary_output = np.zeros_like(m)
    # binary_output = np.copy(img) # Remove this line
    binary_output[(m > mag_thresh[0]) & (m < mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # m = np.sqrt(np.square(x) + np.square(y))
    abs_sobelx = np.abs(x)
    abs_sobely = np.abs(y)
    m = np.arctan(abs_sobely/abs_sobelx)
    binary_output = np.zeros_like(m)
    # print(thresh[0] * 180 / np.pi)
    # print(thresh[1] * 180 / np.pi)
    # print(m)
    binary_output[(m > thresh[0]) & (m < thresh[1])] = 1
    # binary_output = np.copy(img) # Remove this line
    return binary_output

def hls_color_thresh(img, threshLow, threshHigh):
    # 1) Convert to HLS color space
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #Hue (0,180) Light (0,255), satur (0,255)

   
    # 3) Return a binary image of threshold result
    binary_output = np.zeros((img.shape[0], img.shape[1]))
    binary_output[(imgHLS[:,:,0] >= threshLow[0]) & (imgHLS[:,:,0] <= threshHigh[0]) & (imgHLS[:,:,1] >= threshLow[1])  & (imgHLS[:,:,1] <= threshHigh[1])  & (imgHLS[:,:,2] >= threshLow[2]) & (imgHLS[:,:,2] <= threshHigh[2])] = 1
                 
    return binary_output

# def hls_thresh(img, thresh=(100, 255)):
#         """
#         Convert RGB to HLS and threshold to binary image using S channel
#         """
#         hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#         s_channel = hls[:,:,2]
#         binary_output = np.zeros_like(s_channel)
#         binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
#         return binary_output


def combined_thresh(img):
        yellow_low = np.array([0,100,100])
        yellow_high = np.array([50,255,255])

        white_low = np.array([18,0,180])
        white_high = np.array([255,80,255])
        imgThres_yellow = hls_color_thresh(img,yellow_low,yellow_high)
        imgThres_white = hls_color_thresh(img,white_low,white_high)
        abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
        mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
        dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        # hls_bin = hls_thresh(img, thresh=(170, 255))

        combined = np.zeros_like(dir_bin)
        combined[(imgThres_yellow==1) | (imgThres_white==1)] = 1
        combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1)))] = 1

        return combined, abs_bin, mag_bin, dir_bin, imgThres_yellow, imgThres_white

if __name__ == '__main__':
        img_file = 'test_images/straight_lines1.jpg'
        img_file = 'test_images/test5.jpg'

        with open('calibrate_camera.p', 'rb') as f:
                save_dict = pickle.load(f)
        mtx = save_dict['mtx']
        dist = save_dict['dist']

        img = mpimg.imread(img_file)
        img = cv2.undistort(img, mtx, dist, None, mtx)

        combined, abs_bin, mag_bin, dir_bin, imgThres_yellow, imgThres_white = combined_thresh(img)

        plt.subplot(2, 3, 1)
        plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
        plt.subplot(2, 3, 2)
        plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
        plt.subplot(2, 3, 3)
        plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
        # plt.subplot(2, 3, 4)
        # plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
        plt.subplot(2, 3, 5)
        plt.imshow(img)
        plt.subplot(2, 3, 6)
        plt.imshow(combined, cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        plt.show()
