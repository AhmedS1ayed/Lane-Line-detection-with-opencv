#!/usr/bin/env python
# coding: utf-8

# # Generate Binary Threshold

# In[10]:


import matplotlib.pylab as plt
import cv2
import numpy as np

global first_frame
global filter_flag
global debug
debug = 0
first_frame = 1

def binary_threshold(img, low, high):
    output = np.zeros_like(img[:,:,0],  dtype=np.uint8)
    mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0])         & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1])         & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
            
    output[mask] = 1
    return output

def get_binary_image(image):
    global first_frame
    global filter_flag
    
    ### HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]
    L_max, L_mean = np.max(L), np.mean(L)
    print(L_mean, "l_mean")
    S = hls[:,:,2]
    S_max, S_mean = np.max(S), np.mean(S)

    #Changing filter cooficients based on the average lightning
    if (first_frame):
        if ((L_mean < 122) | (L_mean > 126)):
            filter_flag = 0
            print('p0')
        else:
            filter_flag = 1
            print('c0')
            
    
            
    if (not filter_flag):
        #yellow cooficients
        yellow_L_thr = 80
        ye_L_mean_coof = 1.25
        S_max_coof = 0.25
        S_mean_coof = 1.75
        #white cooficients
        white_L_thr = 160
        L_max_coof = 0.8
        wh_L_mean_coof = 1.25
        print('p1')
    else:
        #yellow cooficients
        yellow_L_thr = 0
        ye_L_mean_coof = 1
        S_max_coof = 0.1
        S_mean_coof = 1.7
        #white cooficients
        white_L_thr = 0
        L_max_coof = 0.7
        wh_L_mean_coof = 1.3
        print('c1')
        
        
    # YELLOW
    L_adapt_yellow = max(yellow_L_thr, int(L_mean * ye_L_mean_coof))
    S_adapt_yellow = max(int(S_max * S_max_coof), int(S_mean * S_mean_coof))
    S_adapt_yellow = S_adapt_yellow if (not filter_flag) else 25
    hls_low_yellow = np.array((15, L_adapt_yellow, S_adapt_yellow))
    hls_high_yellow = np.array((30, 255, 255))

    hls_yellow = binary_threshold(hls, hls_low_yellow, hls_high_yellow)

    # WHITE
    L_adapt_white =  max(white_L_thr, int(L_max *L_max_coof),int(L_mean * wh_L_mean_coof))
    hls_low_white = np.array((0, L_adapt_white,  0))
    hls_high_white = np.array((255, 255, 255))

    hls_white = binary_threshold(hls, hls_low_white, hls_high_white)

    
    hls_binary = hls_yellow | hls_white
    
    #show
#     plt.imshow(hls_binary)
    
    return  hls_binary 

# image1 = cv2.imread('straight_lines2.jpg')
# image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  #BGR to RGB   
# hls_binary = get_binary_image(image)
# print(hls_binary.shape)
# plt.imshow(hls_binary)


# # PerspectiveTransform

# In[11]:


def perspectiveTrnsform(srcp ,dstp):
   M = cv2.getPerspectiveTransform(srcp ,dstp)
   Minv = cv2.getPerspectiveTransform(dstp ,srcp)
   return M ,Minv

def warpPerspective(img,imgsize,M):
    return cv2.warpPerspective(img,M,imgsize,cv2.INTER_LINEAR)

global src_first_pt, src_last_pt, dst_first_pt, dst_last_pt
def warp(image):
    global filter_flag
    global src_first_pt, src_last_pt, dst_first_pt, dst_last_pt

    if (not filter_flag):
        srcp = np.float32(
        [[685, 450],
          [1090, 710],
          [220, 710],
          [595, 450]])
        dstp = np.float32(
        [[900, 0],
          [900, 710],
          [250, 710],
          [250, 0]])
        
        src_first_pt = 220
        src_last_pt = 1090
        dst_first_pt = 250
        dst_last_pt = 900
    else:
        srcp = np.array([
            [200,713],
            [630,460],
            [750,460],
            [1200,713]
        ]).astype(np.float32)
        dstp = np.array([
            [300,713],
            [300,0],
            [1100,0],
            [1100,713]
        ]).astype(np.float32)
        
        print('c2')
        
        src_first_pt = 200
        src_last_pt = 1200
        dst_first_pt = 300
        dst_last_pt = 1100
    
    height = image.shape[0]
    width = image.shape[1]
    imgsize2 =(width,height)
    M,Minv = perspectiveTrnsform(srcp ,dstp)
    wraped_img = warpPerspective(image.astype(np.float32),imgsize2,M)
    print("Height:",height,", Width:",width)
    
    #show
#     plt.imshow(wraped_img)
    
    return wraped_img, Minv

# wraped_img, Minv = warp(hls_binary)
# print(wraped_img.shape)
# plt.imshow(wraped_img)


# # Histogram

# In[12]:


def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    #show
#     plt.plot(histogram)
#     plt.savefig('his.png')
    
    return histogram
    
# histogram = get_histogram(wraped_img)
# plt.plot(histogram)


# # sliding window

# In[13]:


def slide_window(binary_warped, histogram):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_high = binary_warped.shape[0] - window*window_height
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    

    # Subplotting two plots 
    f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 4.5))
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    ######################
    plt1.set_title("Sliding window")
    plt1.imshow(out_img)
    plt1.plot(left_fitx, ploty, color='yellow')
    plt1.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
     
    ################################ 
    ## Visualization
    ################################ 
    
    
    out_img2 = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img2)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img2, 1, window_img, 0.3, 0)
    
    plt2.set_title("Lanes")
    plt2.imshow(result)
    plt2.plot(left_fitx, ploty, color='yellow')
    plt2.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('lanes.jpg')
    
    
    
    #################
    
    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty
    
    return ret, out_img, result
    

# draw_info, out_img, lanes = slide_window(wraped_img, histogram)


# # Lane Curvature

# In[14]:


def measure_curvature(lines_info):
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/700 

    ploty = lines_info['ploty']
    leftx = lines_info['left_fitx']
    rightx = lines_info['right_fitx']

    leftx = leftx[::-1]  
    rightx = rightx[::-1]  

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    print("left_curverad:",left_curverad, 'm',", right_curverad:", right_curverad, 'm')
    
    return left_curverad, right_curverad
    
# left_curverad, right_curverad = measure_curvature(draw_info)


# # Draw Lane Lines

# In[15]:


def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']
    
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # FIll the lane 
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (255,255, 0))
    
    # Draw left lane line
    pts_left_ = np.array([np.transpose(np.vstack([left_fitx -7, ploty]))])
    pts_left__ = np.array([np.flipud(np.transpose(np.vstack([left_fitx + 7, ploty])))])
    pts_l = np.hstack((pts_left_, pts_left__))
    cv2.fillPoly(color_warp, np.int_([pts_l]), (255,0, 0))

    # Draw right lane line
    pts_right_ = np.array([np.transpose(np.vstack([right_fitx -7, ploty]))])
    pts_right__ = np.array([np.flipud(np.transpose(np.vstack([right_fitx + 7, ploty])))])
    pts_r = np.hstack((pts_right_, pts_right__))
    cv2.fillPoly(color_warp, np.int_([pts_r]), (0,0, 255))
    
    # Take inverse prespective
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    
    #show
#     plt.imshow(result)
    
    return result


# result = draw_lane_lines(image, wraped_img, Minv, draw_info)
# plt.imshow(result)


# # Proccess Image

# In[17]:


global used_warped
global used_draw_info

first_frame = 1

def single_pt_perspective(x, src_first_pt, src_last_pt, dst_first_pt, dst_last_pt):
#     new_x = (s_w/d_w)*(x-d_f) + s_f
    new_x = ((src_last_pt-src_first_pt)/(dst_last_pt-dst_first_pt)) * (x-dst_first_pt) + src_first_pt
    return new_x

def process_image(image):
    global used_warped
    global used_draw_info
    global filter_flag
    
#     if (filter_flag):
        
#         image = image1
#         print('k')

    # Generating HLS Binary Threshold
    hls_binary = get_binary_image(image)
#     plt.imshow(hls_binary)
    
    # Perspective Transform
    wraped_img, Minv = warp(hls_binary)
#     plt.imshow(wraped_img)
  
    # Getting Histogram
    histogram = get_histogram(wraped_img)
#     plt.savefig(histogram, format='png')
  
    # Sliding Window to detect lane lines
    try:
        draw_info, out_img, lanes = slide_window(wraped_img, histogram)
    except:
        draw_info = used_draw_info
        
    used_draw_info = draw_info
    
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']
    
#     plt.savefig(img_buf, format='png')
   
    # Measuring Curvature
    left_curverad, right_curverad = measure_curvature(draw_info)
   
    
    # Visualizing Lane Lines Info
    result = draw_lane_lines(image, wraped_img, Minv, draw_info)
#     plt.imshow(result)
  
    
    # Annotating curvature 
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    curvature_text = 'The radius of curvature = ' + str(round(left_curverad, 3)) + 'm'
    cv2.putText(result, curvature_text, (30, 60), fontType, 1.5, (255, 255, 255), 3)
   
    # Annotating deviation
    global src_first_pt, src_last_pt, dst_first_pt, dst_last_pt
#     x = (s_w/d_w)*(y-d_f) + s_f
    right_fitx_inv = single_pt_perspective(draw_info['right_fitx'][-1], src_first_pt, src_last_pt, dst_first_pt, dst_last_pt)
    left_fitx_inv = single_pt_perspective(draw_info['left_fitx'][-1], src_first_pt, src_last_pt, dst_first_pt, dst_last_pt)
    
    deviation_pixels = image.shape[1]/2 - abs(right_fitx_inv/2 + left_fitx_inv/2)
    print(right_fitx_inv)
    print(left_fitx_inv)
    print(image.shape[1]/2)
    
#     xm_per_pix = 3.7/700 
    xm_per_pix = 6.77/1280 
    deviation = deviation_pixels * xm_per_pix
    direction = "left" if deviation < 0 else "right"
    deviation_text = 'Vehicle is ' + str(round(abs(deviation), 3)) + 'm ' + direction + ' of center'
    cv2.putText(result, deviation_text, (30, 110), fontType, 1.5, (255, 255, 255), 3)
    
    used_warped = wraped_img
    used_draw_info = draw_info
    first_frame = 0
    
    global debug
#     debug = 1
    if debug == 1:
        
        resized = cv2.resize(result, (2560,1440))
#         resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        img1 = np.dstack((hls_binary, hls_binary, hls_binary))*255
        img2 = np.dstack((wraped_img, wraped_img, wraped_img))*255
        lanes = cv2.imread('lanes.jpg')
        lanes_ = cv2.resize(lanes, (3840,1440))
        
        v_img = np.vstack((img1, img2))
        h_img = np.hstack((resized, v_img))
        vv_img = np.vstack((h_img, lanes_))

#         cv2.imwrite(r'vvv.png',vv_img)
        result = vv_img
    
    return result 

# image1 = cv2.imread('straight_lines2.jpg')
# image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  #BGR to RGB   
# result_image = process_image(image)
# plt.imshow(result_image)


# # Create Output video

# In[1]:


import imageio
#imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython import get_ipython
def create_video(clip,output):    
# output = 'pv1.mp4'
# clip = VideoFileClip("project_video.mp4")
    video_clip = clip.fl_image(process_image)
    get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')


# # Main

# In[ ]:


import sys
 
def main(argv):
    inputfile = ''
    outputfile = ''
#     try:
#         opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#     except getopt.GetoptError:
#         print ('test.py -i <inputfile> -o <outputfile>')
#         sys.exit(1)
 
 
 
    for arg in sys.argv:
        if arg == '-h':
            print ('test.py <inputfile> <outputfile> <--d>') #--d for debug
#             sys.exit(1)
        elif arg == '--d':
            global debug
            debug = 1
            print('d')
            
    n = len(sys.argv)
    try:
        for i in range(0, n):
            inputfile = sys.argv[1]
            outputfile = sys.argv[2]
        print ('Input file is "', inputfile)
        print ('Output file is "', outputfile)
    except:
        print ('test.py <inputfile> <outputfile> <--d>') #--d for debug
 
    
    clip = VideoFileClip(inputfile)
    create_video(clip,outputfile)
    
if __name__ == "__main__":
    main(sys.argv[1:])

