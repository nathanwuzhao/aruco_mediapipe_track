import cv2 as cv #4.5.5.62
import numpy as np #1.26.4
import math
import time
from scipy.spatial.transform import Rotation as R

#dictionary for storing which tags associate to which finger
finger_tag_ids = {
    'LH':{ #left hand, tag order corresponds to proximal, distal 
        'thumb': [11, 11, 11],
        'index': [11, 2, 0],
        'middle': [11, 7, 10],
        'ring': [11, 11, 11],
        'pinky': [11, 11, 11]
    },
    'RH':{ #right hand, tag order corresponds to proximal, distal
        'thumb': [11, 11, 11],
        'index': [11, 11, 11],
        'middle': [11, 11, 11],
        'ring': [11, 11, 11],
        'pinky': [11, 11, 11]
    }
}

# joint definitions, each joint is between two adjacent markers
joint_definitions = {
    'LH':{
        'thumb': [(11, 11), (11,11)], #MCP, IP
        'index': [(11, 2), (2, 0)], #MCP, PIP (DIP inferred from PIP?)
        'middle': [(11, 7), (7, 10)], #MCP, PIP (DIP inferred from PIP?)
        'ring': [(11, 11), (11, 11)], #MCP, PIP (DIP inferred from PIP?)
        'pinky': [(11, 11), (11, 11)] #MCP, PIP (DIP inferred from PIP?)
    },
    'RH':{ #right hand, tag order corresponds to proximal, distal
        'thumb': [(11, 11), (11,11)], #MCP, IP
        'index': [(11, 11), (11, 11)], #MCP, PIP (DIP inferred from PIP?)
        'middle': [(11, 11), (11, 11)], #MCP, PIP (DIP inferred from PIP?)
        'ring': [(11, 11), (11, 11)], #MCP, PIP (DIP inferred from PIP?)
        'pinky': [(11, 11), (11, 11)] #MCP, PIP (DIP inferred from PIP?)
    }
}

# Initialize data structures
hand_data = {'LH': {}, 'RH':{}}
joint_angles = {'LH': {}, 'RH': {}}

# Initialize hand_data based on finger_tag_ids (marker-centric)
for hand in finger_tag_ids:
    for finger in finger_tag_ids[hand]:
        hand_data[hand][finger] = {}
        for tag_id in finger_tag_ids[hand][finger]:
            hand_data[hand][finger][tag_id] = {
                'id': tag_id, 
                'tvec': None,
                'rvec': None,
                'rotation_matrix': None,
                'euler': None,
                'detected': False
            }

# Initialize joint_angles based on joint_definitions (joint-centric)
for hand in joint_definitions:
    joint_angles[hand] = {}
    for finger in joint_definitions[hand]:
        joint_angles[hand][finger] = {}
        for i, (proximal_id, distal_id) in enumerate(joint_definitions[hand][finger]):
            joint_name = f"joint_{i}"
            joint_angles[hand][finger][joint_name] = None

#aruco detection stuff
aruco_marker_side_length = 10 #mm
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50) #declare aruco tag dictionary
aruco_parameters = cv.aruco.DetectorParameters_create() #detector parameters

#camera calibration
calibration_filename = 'calibration_chessboard.yaml' #filename for calibration data
cv_file = cv.FileStorage(calibration_filename, cv.FILE_STORAGE_READ) #read yaml file
mtx = cv_file.getNode('K').mat() #get camera matrix
dst = cv_file.getNode('D').mat() #get camera distortion coefficients
cv_file.release()

euler_order = 'yxz'

def smooth_angle_sign(prev_angle, new_angle, threshold=20):
    #smoothing function to try to get rid of angle flipping bullshit
    if prev_angle is None:
        return new_angle
    if abs(abs(prev_angle) - abs(new_angle)) < threshold and np.sign(prev_angle) != np.sign(new_angle):
        return prev_angle  # reject sudden sign flip
    return new_angle

def calculate_joint_angle_orient_mat(R1, R2, joint_axis='y'): 
    #calculate relative rotation using transpose
    R_rel= np.dot(R2, R1.T)

    #convert to scipy Rotation object
    r_rel = R.from_matrix(R_rel)

    # extract euler angles - order matters for joint interpretation
    if joint_axis == 'y':
        # Extract y-axis rotation (flexion/extension)
        euler = r_rel.as_euler(euler_order, degrees=True)
        joint_angle = euler[0]  # y-axis rotation
    elif joint_axis == 'x':
        euler = r_rel.as_euler(euler_order, degrees=True)
        joint_angle = euler[1]  # x-axis rotation
    elif joint_axis == 'z':
        euler = r_rel.as_euler(euler_order, degrees=True)
        joint_angle = euler[2]  # z-axis rotation
    else:
        # Default to y-axis
        euler = r_rel.as_euler(euler_order, degrees=True)
        joint_angle = euler[1]
    
    return joint_angle

def calculate_joint_angle_orient_y(R1, R2, reference_vector=None):
    #θ = signum((OXprx × OXdis) · Vref) * arccos(OXprx · OXdis)
    
    #get y vectors from input rotation matrices
    OXprx = R1[:, 1] #y axis of proximal marker
    OXdis = R2[:, 1] #y axis of distal marker

    #normalize y vectors
    OXprx = OXprx / np.linalg.norm(OXprx)
    OXdis = OXdis / np.linalg.norm(OXdis)


    #default reference vector (z axis, up)
    if reference_vector is None:
        reference_vector = np.array([0, 0, 1])

    #normalize (unit vector)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    #calculate dot product (angle magnitude)
    dot_product = np.dot(OXprx, OXdis)
    #clamp to [-1, 1], avoid numerical errors in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    #angle magnitude
    angle_magnitude = np.arccos(dot_product)

    #cross product because thats what the formula says idk whats going on i slept through this unit in calc3
    cross_product = np.cross(OXprx, OXdis)
    #calculate sign using reference vector 
    sign = np.sign(np.dot(cross_product, reference_vector))

    #final angle, convert to degrees
    joint_angle = sign * angle_magnitude
    joint_angle_degrees = np.degrees(joint_angle)

    return joint_angle_degrees

def calculate_all_joint_angles(method='orient_mat'):
    #calculate for all finger joints
    for hand in joint_definitions:
        for finger in joint_definitions[hand]:
            joint_pairs = joint_definitions[hand][finger]
            for i, (proximal_id, distal_id) in enumerate(joint_pairs):
                joint_name = f"joint_{i}"

                #check if both tags are detected
                if(hand_data[hand][finger][proximal_id]['detected'] and 
                   hand_data[hand][finger][distal_id]['detected']):
                    
                    #get rotation matrices
                    R1 = hand_data[hand][finger][proximal_id]['rotation_matrix']
                    R2 = hand_data[hand][finger][distal_id]['rotation_matrix']

                    if R1 is not None and R2 is not None:
                        if method == 'orient_y':
                            angle = calculate_joint_angle_orient_y(R1, R2, reference_vector=R1[:, 2])
                        else:
                            angle = calculate_joint_angle_orient_mat(R1, R2, 'x') #default to orient mat

                        #prev = joint_angles[hand][finger][joint_name]
                        joint_angles[hand][finger][joint_name] = abs(angle)
                    else:
                        joint_angles[hand][finger][joint_name] = None
                else:
                    joint_angles[hand][finger][joint_name] = None

def update_marker_data(marker_id, rvec, tvec, rotation_matrix, euler):
    for hand in finger_tag_ids:
        for finger in finger_tag_ids[hand]:
            if marker_id in finger_tag_ids[hand][finger]:
                hand_data[hand][finger][marker_id]['tvec'] = tvec
                hand_data[hand][finger][marker_id]['rvec'] = rvec
                hand_data[hand][finger][marker_id]['rotation_matrix'] = rotation_matrix
                hand_data[hand][finger][marker_id]['euler'] = euler
                hand_data[hand][finger][marker_id]['detected'] = True
                return
            
def reset_detection_flags():
    for hand in hand_data:
        for finger in hand_data[hand]:
            for marker_id in hand_data[hand][finger]:
                hand_data[hand][finger][marker_id]['detected'] = False

def display_joint_angles(frame):
    #add offset for every new angle shown
    y_offset = 30
    for hand in joint_angles:
        for finger in joint_angles[hand]:
            for joint_name, angle in joint_angles[hand][finger].items():
                if angle is not None:
                    text = f"{hand}_{finger}_{joint_name}: {angle:.1f} degrees"
                    cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20

def pose_estimation(frame, matrix_coefficients, distortion_coefficients):

    reset_detection_flags()

    #convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #detect markers
    (corners, ids, rejected) = cv.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    
    #markers detected
    if ids is not None:
        #draw markers with corners and marker ids 
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        #process each detected marker
        for i in range(0, len(ids)):
            marker_id = ids[i][0]

            #estimate vectors/6D pose; function is deprecated after opencv 4.6, too lazy to figure out solvePnP
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], aruco_marker_side_length, matrix_coefficients, distortion_coefficients)

            #draw them jawns
            cv.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 8)

            #ravel tvec to get as (3,) array, get translation vectors
            #transform_translation_x, transform_translation_y, transform_translation_z = tvec.ravel()
            
            #reshape this jawn
            rvec = rvec.reshape(3, 1)
            #convert rvec to 3x3 rotation matrix
            output, jacobian = cv.Rodrigues(rvec)
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = output

            #scipy convert this jawn to euler angles
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            euler_angles = r.as_euler(euler_order, degrees=True) #pitch_Y, yaw_Z, roll_x in degrees
            
            #update this jawn
            update_marker_data(marker_id=marker_id, rvec=rvec, tvec=tvec, rotation_matrix=output, euler=euler_angles)
        
        #calculate joint angles after processing all markers
        calculate_all_joint_angles(method='orient_mat')
        
        display_joint_angles(frame)
            
    return frame

def main(index):
    #index this jawn so we can do multiple 
    capture = cv.VideoCapture(index)
    if not capture.isOpened(): #could not open
        print('could not open video source {}'.format(index))
        exit() #quit

    while True:
        ret, frame = capture.read()
        if not ret: #could not read
            print('could not read from video source {}'.format(0))
            break #quit

        #call pose estimation function, show aruco tags with pose estimation
        output = pose_estimation(frame, mtx, dst)
        cv.imshow("capture{}".format(index), output)

        if cv.waitKey(1) == ord('q'): #exit if q pressed
            break
    
    capture.release()

if __name__ == '__main__':
    #call main on video source 0
    main(0)
    cv.destroyAllWindows()