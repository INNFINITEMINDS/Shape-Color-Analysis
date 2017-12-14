#classes and subclasses to import
import cv2
import datetime
import numpy as np
import os

filename = 'OutputImage.csv'
#################################################################################################
# DO NOT EDIT!!!
#################################################################################################
#subroutine to write results to a csv
def writecsv(color,shape,(cx,cy)):
    global filename
    #open csv file in append mode
    filep = open(filename,'a')
    # create string data to write per image
    datastr = "," + color + "-" + shape + "-" + str(cx) + "-" + str(cy)
    #write to csv
    filep.write(datastr)

#################################################################################################
# DO NOT EDIT!!!
#################################################################################################
def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def main(video_file_with_path):	
	cap = cv2.VideoCapture(video_file_with_path)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	image_red = cv2.imread("yellow_flower.png",-1)
	image_blue = cv2.imread("pink_flower.png",-1)
	image_green = cv2.imread("red_flower.png",-1)
	print cap.isOpened()
	print cap.get(cv2.CAP_PROP_FPS)
	cap.set(cv2.CAP_PROP_POS_FRAMES,1)
	cap.grab()
	ret, frame = cap.read()
	count = 0
	start_time = datetime.datetime.now()
	list=[]
	while(ret):
	    ret2, frame = cap.read()
	    if(ret2):
		if(count==100):
		    cv2.imwrite("frame.jpg",frame)
		    #break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		ret, thresh = cv2.threshold(blurred, 127, 255, 1)
		
		_, contours, hierarchy = cv2.findContours(thresh, 1, 2)
		

		for cnt in contours:
		    M = cv2.moments(cnt)
		    cx = int(M['m10'] / M['m00'])
		    cy = int(M['m01'] / M['m00'])
		    if cx not in list:
		        list.append(cx)
		        color = frame[cy, cx]                                           
		        if (color[0] >= 0 and color[0] <= 10 and color[1] >= 0 and color[1] <= 10 and color[2] >= 250 and color[2] <= 255):  # logic for red
		            colour = "r"
		            x, y, w, h = cv2.boundingRect(cnt)
		            overlay_image = cv2.resize(image_red, (h, w))
		        if (color[0] >= 250 and color[0] <= 255 and color[1] >= 0 and color[1] <= 10 and color[2] >= 0 and color[2] <= 10):  # logic for blue
		            colour = "b"
		            x, y, w, h = cv2.boundingRect(cnt)
		            overlay_image = cv2.resize(image_blue, (h, w))
		        if (color[0] >= 0 and color[0] <= 10 and color[1] >= 120 and color[1] <= 130 and color[2] >= 0 and color[2] <= 10):  # logic for green
		            colour = "g"
		            x, y, w, h = cv2.boundingRect(cnt)
		            overlay_image = cv2.resize(image_green, (h, w))
		              
		    frame[y:y + w, x:x + h, :] = blend_transparent(frame[y:y + w, x:x + h, :], overlay_image)
	 

		cv2.imshow('frame',frame)
		out.write(frame)
		cv2.waitKey(40)
		count+=1
	    else:
		break

	print count
	cap.release()
	end_time = datetime.datetime.now()
	tp = end_time-start_time
	print tp

#################################################################################

#####################################################################################################
    #sample of overlay code for each frame
    #x,y,w,h = cv2.boundingRect(current_contour)
    #overlay_image = cv2.resize(image_red,(h,w))
    #frame[y:y+w,x:x+h,:] = blend_transparent(frame[y:y+w,x:x+h,:], overlay_image)
#######################################################################################################

#################################################################################################
# DO NOT EDIT!!!
#################################################################################################
#main where the path is set for the directory containing the test images
if __name__ == "__main__":
    main('./Video.mp4')
