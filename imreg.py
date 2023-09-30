import cv2 
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

def extract(img): 
    #get feats
    feats = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,0.01,3)
    
    #get kps 
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
    des = orb.compute(img, kps)
    
    #ret them kps
    return kps, des 

def detect_edges(img): 
    # Assuming 'image' is your input image and 't1' and 't2' are your Canny thresholds.
    edges = cv2.Canny(img, threshold1=15, threshold2=45)

    # Create a 3x3 matrix filled with ones. This will be used as a kernel for dilation and erosion operations.
    horizontal_kernel = np.ones((2,2), np.uint8)
    
    # Dilate the edges. Dilation adds pixels to the boundaries of objects in an image.
    dilated_edges = cv2.dilate(edges, horizontal_kernel, iterations=1)

    # Erode the dilated edges. Erosion removes pixels from the boundaries of objects in an image.
    # This can help to remove noise and small unwanted details.
    eroded_edges = cv2.erode(dilated_edges, horizontal_kernel, iterations=1)

    # Find the contours in the eroded edges image. Contours are simply the boundaries of the connected objects.
    # The function returns a list of contours and a hierarchy (which is not used here, hence the underscore).
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image (all zeros) with the same shape as the eroded edges image.
    # This will be used to draw the filtered contours.
    filtered_edges = np.zeros_like(dilated_edges)

    # Loop through each detected contour.
    for contour in contours:
        # Check if the length (perimeter) of the contour is greater than a threshold (100 in this case).
        if cv2.arcLength(contour, True) > 90:
            # Draw the contour on the filtered_edges image. The contour is filled with white (255).
            cv2.drawContours(filtered_edges, [contour], -1, (255), thickness=cv2.FILLED)
    
    #set these just so they exists
    rightmost_point = (0,filtered_edges.shape[0])
    leftmost_point = (0,0)

    circles = cv2.HoughCircles(filtered_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=90, param1=20, param2=10, minRadius=15, maxRadius=80)

    ret_filtered_edges = cv2.cvtColor(filtered_edges, cv2.COLOR_GRAY2BGR)
    center_difference = 0
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        
        # Sort circles based on x-coordinates
        sorted_circles = sorted(circles[0, :], key=lambda x: x[0])

        # Get the leftmost and rightmost circles
        leftmost_circle = sorted_circles[0]
        rightmost_circle = sorted_circles[-1]

        for i in circles[0, :]:
            # Draw the outer circle on the temporary image
            cv2.circle(ret_filtered_edges, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # Draw the center of the circle on the temporary image
            cv2.circle(ret_filtered_edges, (i[0], i[1]), 2, (0, 255, 255), 1)

        leftmost_point = (leftmost_circle[0] + leftmost_circle[2], leftmost_circle[1])
        rightmost_point = (rightmost_circle[0] - rightmost_circle[2], rightmost_circle[1])

        #calculate difference
        center_difference = rightmost_point[0] - leftmost_point[0]
        if (center_difference < 10 or center_difference > 200): 
            center_difference = -1
        # Draw vertical lines on the edges of the leftmost and rightmost circles
        cv2.line(ret_filtered_edges, (leftmost_point[0], filtered_edges.shape[0]//3), (leftmost_point[0], 2*(filtered_edges.shape[0]//3)), (255, 0, 0), 2)
        cv2.line(ret_filtered_edges, (rightmost_point[0], filtered_edges.shape[0]//3), (rightmost_point[0], 2*(filtered_edges.shape[0]//3)), (255, 0, 0), 2)


    img_top = filtered_edges[:filtered_edges.shape[0]//2, :]
    img_bottom = filtered_edges[filtered_edges.shape[0]//2:filtered_edges.shape[0], :]

    top_difference, first_column_with_one_top, last_column_with_one_top = calculate_width(img_top)
    bottom_difference, first_column_with_one_bottom, last_column_with_one_bottom = calculate_width(img_bottom)

    cv2.line(ret_filtered_edges, (first_column_with_one_top, 0),
             (first_column_with_one_top, filtered_edges.shape[0]//2), (0, 0, 255), 1)
    cv2.line(ret_filtered_edges, (last_column_with_one_top, 0),
             (last_column_with_one_top, filtered_edges.shape[0]//2), (0, 0, 255), 1)

    cv2.line(ret_filtered_edges, (first_column_with_one_bottom, filtered_edges.shape[0]//2),
             (first_column_with_one_bottom, filtered_edges.shape[0]), (0,255,255), 1)
    cv2.line(ret_filtered_edges, (last_column_with_one_bottom, filtered_edges.shape[0]//2),
             (last_column_with_one_bottom, filtered_edges.shape[0]), (0,255,255), 1)

    return ret_filtered_edges, top_difference, bottom_difference, center_difference

def calculate_width(img):
    # Binarize the *_edges matrix
    binary_matrix = np.where(img > 0, 1, 0)
    count = 0

    # double check if anything found
    for i in range(binary_matrix.shape[0]):
        for j in range(binary_matrix.shape[1]):
            if binary_matrix[i][j] == 1:
                count += 1

    if (count==0):
        return 0,0,0

    # Find the first and last columns that contain a '1'
    first_column_with_one = np.where(binary_matrix.any(axis=0))[0][0]
    last_column_with_one = np.where(binary_matrix.any(axis=0))[0][-1]

    return last_column_with_one - first_column_with_one, first_column_with_one, last_column_with_one

def process_frame(img): 
    global roiselected,left,right,top,bottom

    #load image
    orig = cv2.imread(img) 


    if not roiselected: 
        r = get_roi(orig)
        roiselected = True
        top = int(r[1])
        bottom = int(r[1]+r[3])
        left = int(r[0])
        right = int(r[0]+r[2])


    orig = orig[top:bottom, left:right]
    #extract kps 
    #kps, des = extract(orig)

    #extract edges
    filtered, top_difference, bottom_difference, center_difference = detect_edges(orig)
    
    print(top_difference, bottom_difference, center_difference)
    #label kps in image
    #for p in kps:
        #x,y = map(lambda x: int(round(x)), p.pt)
        #cv2.circle(filtered, (x,y), color=(0,255,0), radius=2)  
    
    #show image
    cv2.imshow("Features", filtered)
    cv2.waitKey(10) 

    return top_difference, bottom_difference, center_difference


def get_roi(img):
    # Convert image to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use selectROI to get the rectangle coordinates
    r = cv2.selectROI(img_rgb)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return r
#END roi 

def main(data): 
    global top_values, bottom_values, center_values
    for image in data:    
        if (image.endswith(".BMP")):
            top_difference, bottom_difference, center_difference = process_frame(folder_dir +"/"+ image)
            if (top_difference < 20): 
                top_values.append(top_values[-1])
            else: 
                top_values.append(top_difference)
            if (bottom_difference < 5):
                bottom_values.append(bottom_values[-1])
            else:
                bottom_values.append(bottom_difference)
            if (center_difference < 5): 
                center_values.append(center_values[-1])
            else:
                center_values.append(center_difference)
            plt.plot(top_values, color="red", label="Top Diffs")
            plt.plot(bottom_values, color="yellow", label="Bottom Diffs")
            plt.plot(center_values, color="blue", label="Center Diffs")
            plt.pause(0.0001)

rootdir = "/home/binaryblaze/Desktop/KasperiP"
roiselected = False
left = 0
right = 0
top = 0
bottom = 0 
top_values = [0]
bottom_values = [0]
center_values = [0]
for root, subFolders, files in os.walk(rootdir):
    for folder in subFolders:
        roiselected = False
        folder_dir = rootdir + "/" + folder
        content = os.listdir(folder_dir)
        content.sort()
        orb = cv2.ORB_create()
        main(content)
        top_values = [0]
        bottom_values = [0]
        center_values = [0]