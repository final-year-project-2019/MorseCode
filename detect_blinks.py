import cv2
import time
import dlib
import argparse
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from scipy.spatial import distance

def calculateEar(eye):
    """Summary

    EAR is a ratio of the vertical to the horizontal distance of the eye. This function is used to calculate the EAR value for an eye given the coordinates of the eye as an argument
    
    Parameters:
    eye (float[]): contains the coordinates for the eye.

    Returns:
    ear (float): eye aspect ratio 
    """

    #calculate the vertical distance between the eye for two pairs of points
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    
    #calculate the horizontal distance between the eye for a pair of points
    C = distance.euclidean(eye[0],eye[3])

    #calculation of EAR
    ear = ( A + B )/2.0*C;

    return ear


#For parsing both of the arguments of the program, which are paths to the landmarking dataset and the video being evaluated
parser = argparse.ArgumentParser(description="This takes the pre-trained landmarkind dataset and the input video stream")
parser.add_argument("-p","--shape-predictor",required=True,help="path to faical landmark predictor")
parser.add_argument("-v","--video",type=str,default="",help="path to video being evaluated")
args = vars(parser.parse_args())

LCOUNTER = 0 #counter for the left eye
RCOUNTER = 0 #counter for the right eye

#loading dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"]) #loaeding the pretrained dataset using the argument to its path

#getting the indexes of the landmarks for the left and right eye
(lStart, lEnds) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnds) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0)

while True:
    # read from the video source
    ret,frame = vs.read() # resize the window in which the stream is displayed
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converting image to grayscale
    cv2.imshow('frame',gray)
    rects = detector(gray,0) # detecting faces in the image
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnds]
        rightEye = shape[rStart:rEnds]
        leftEAR = calculateEar(leftEye)
        rightEAR = calculateEar(rightEye)
        print("Left EAR: ", leftEAR)
        print("rightEAR: ", rightEAR)
        time.sleep(1.0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vs.release()
cv2.destroyAllWindows()