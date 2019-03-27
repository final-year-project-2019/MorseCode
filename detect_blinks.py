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

print(calculateEar([1,2,3,4,5,6]))