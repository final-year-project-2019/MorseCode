from scipy.spatial import distance
import argparse

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
parser.add_argument("-v","--video",required=True,type=str,default="",help="path to video being evaluated")
args = vars(parser.parse_args())


print(calculateEar([1,2,3,4,5,6]))