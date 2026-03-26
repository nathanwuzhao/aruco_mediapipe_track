import cv2 as cv #4.5.5.62
import numpy as np #1.26.4
import math
import time
import threading
from collections import deque
from scipy.spatial.transform import Rotation as R

#global quit flag for camera setup
quit_flag = False

#frame synchronization stuff
cache_length = 5 #number of frames to cache
frame_cache= {'cam0': deque(maxlen=cache_length), 'cam1': deque(maxlen=cache_length)}
cache_lock = threading.Lock()
sync_event = threading.Event()
frames_ready = {'cam0': False, 'cam1': False}

# joint definitions dictionary with all marker assignments. each joint is between two adjacent markers. 
# id 11 is used as a placeholder
joint_definitions = {
    'LH':{
        'thumb': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'IP', 'dof':1}, #flexion only
        ], 
        'index': [
            {'proximal': 11, 'distal': 2, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 2, 'distal': 0, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'middle': [
            {'proximal': 11, 'distal': 7, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 7, 'distal': 10, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'ring': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'pinky': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ]
    },
    'RH':{ 
        'thumb': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'IP', 'dof':1}, #flexion only
        ], 
        'index': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'middle': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'ring': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ], 
        'pinky': [
            {'proximal': 11, 'distal': 11, 'type': 'MCP', 'dof':2}, #flexion + abduction
            {'proximal': 11, 'distal': 11, 'type': 'PIP', 'dof':1}, #flexion only
        ]
    }
}

#extract all unique marker ids from joint definitions
def get_all_marker_ids():
    marker_ids = set()
    for hand in joint_definitions:
        for finger in joint_definitions[hand]:
            for joint_def in joint_definitions[hand][finger]:
                marker_ids.add(joint_def['proximal'])
                marker_ids.add(joint_def['distal'])
    
    return marker_ids

#initialize, with structure for 2dof
def initialize_joint_angles():
    joint_angles = {'LH': {}, 'RH': {}}

    for hand in joint_definitions:
        joint_angles[hand] = {}
        for finger in joint_definitions[hand]:
            joint_angles[hand][finger] = {}
            for i, joint_def in enumerate(joint_definitions[hand][finger]):
                joint_name = f"{joint_def['type']}" #use joint type as name

                if joint_def['dof'] == 1:
                    joint_angles[hand][finger][joint_name] = None #store as single value
                else: 
                    joint_angles[hand][finger][joint_name] = { #store as dictionary
                        'flexion': None,
                        'abduction': None 
                    }
    return joint_angles

#initialize these jawns
joint_angles = initialize_joint_angles()
hand_data = {'LH': {}, 'RH': {}}

for hand in joint_definitions:
    for finger in joint_definitions[hand]:
        hand_data[hand][finger] = {}
        #get unique marker ids for this finger from joint defs
        finger_markers = set()
        for joint_def in joint_definitions[hand][finger]:
            finger_markers.add(joint_def['proximal'])
            finger_markers.add(joint_def['distal'])
        
        # initialize each marker
        for marker_id in finger_markers:
            hand_data[hand][finger][marker_id] = {
                'id': marker_id, 
                'tvec': None,
                'rvec': None,
                'rotation_matrix': None,
                'euler': None,
                'detected': False
            }

#aruco detection stuff
aruco_marker_side_length = 9 #mm
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50) #declare aruco tag dictionary
aruco_parameters = cv.aruco.DetectorParameters_create() #detector parameters

#camera calibration
calibration_filename = 'calibration_chessboard.yaml' #filename for calibration data
cv_file = cv.FileStorage(calibration_filename, cv.FILE_STORAGE_READ) #read yaml file
mtx = cv_file.getNode('K').mat() #get camera matrix
dst = cv_file.getNode('D').mat() #get camera distortion coefficients
cv_file.release()

#angle stuff
euler_order = 'yxz'
flexion_threshold = 8.0 #degrees
flexion_axis_default = 'y'
abduction_axis_default = 'x'

def apply_angle_flexion_threshold(angle, threshold=flexion_threshold):
    if angle is None:
        return None
    
    abs_angle = abs(angle)
    return 0.0 if abs_angle < threshold else abs_angle

def calculate_joint_angle_orient_mat(R1, R2, axis='y'): 
    #calculate relative rotation using transpose
    R_rel= np.dot(R2, R1.T)

    #convert to scipy Rotation object
    r_rel = R.from_matrix(R_rel)
    
    #extract euler angles
    euler = r_rel.as_euler(euler_order, degrees=True)
    axis_map = {'y': 0, 'x': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 1) #default to x
    
    return euler[axis_idx]

def calculate_joint_angle_orient_axis(R1, R2, axis='y', reference_vector=None):
    #θ = signum((OXprx × OXdis) · Vref) * arccos(OXprx · OXdis)
    
    # Select axis vectors based on parameter
    if axis.lower() == 'x':
        OXprx = R1[:, 0]  # x axis of proximal marker
        OXdis = R2[:, 0]  # x axis of distal marker
        default_ref = np.array([0, 1, 0])  # y axis as default reference
    elif axis.lower() == 'y':
        OXprx = R1[:, 1]  # y axis of proximal marker
        OXdis = R2[:, 1]  # y axis of distal marker
        default_ref = np.array([0, 0, 1])  # z axis as default reference
    elif axis.lower() == 'z':
        OXprx = R1[:, 2]  # z axis of proximal marker
        OXdis = R2[:, 2]  # z axis of distal marker
        default_ref = np.array([1, 0, 0])  # x axis as default reference
    else:
        # Default to y-axis if invalid parameter
        OXprx = R1[:, 1]
        OXdis = R2[:, 1] 
        default_ref = np.array([0, 0, 1])

    #normalize y vectors
    OXprx = OXprx / np.linalg.norm(OXprx)
    OXdis = OXdis / np.linalg.norm(OXdis)

    #default reference vector (z axis, up)
    if reference_vector is None:
        reference_vector = default_ref

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
    joint_angle = sign * np.degrees(angle_magnitude)

    return joint_angle

def calculate_joint_angles(R1, R2, joint_dof, method='orient_axis', flexion_axis='y', abduction_axis='x'):
    if joint_dof == '1':
        if method == 'orient_axis':
            #use z of proximal marker
            reference = R1[:,2]
            angle = calculate_joint_angle_orient_axis(R1, R2, axis=flexion_axis, reference_vector=reference)
        else:
            #orient matrix method
            angle = calculate_joint_angle_orient_mat(R1, R2, axis=flexion_axis)

        return apply_angle_flexion_threshold(angle)
    
    elif joint_dof == '2':
        angles = {}

        if method == 'orient_axis':
            #use z of proximal marker
            reference = R1[:,2]    
            flexion_angle = calculate_joint_angle_orient_axis(R1, R2, axis=flexion_axis, reference_vector=reference)
            angles['flexion'] = apply_angle_flexion_threshold(flexion_angle)

            abduction_angle = calculate_joint_angle_orient_axis(R1, R2, axis=abduction_axis, reference_vector=reference)
            angles['abduction'] = abs(abduction_angle) if abduction_angle is not None else None

        else:
            #uhhh relative rotation thing agian
            R_rel = np.dot(R2, R1.T)
            r_rel = R.from_matrix(R_rel)
            euler = r_rel.as_euler('yxz', degrees=True)
            
            #map axes to euler index
            flexion_idx = {'y': 0, 'x': 1, 'z': 2}.get(flexion_axis, 0)
            abduction_idx = {'y': 0, 'x': 1, 'z': 2}.get(abduction_axis, 1)
            
            # apply threshold to flexion, but not abduction
            angles['flexion'] = apply_angle_flexion_threshold(euler[flexion_idx])
            angles['abduction'] = abs(euler[abduction_idx])

    else: 
        return None

def calculate_all_joint_angles(method='orient_mat', flexion_axis='y', abduction_axis='x'):
    #calculate for all finger joints
    for hand in joint_definitions:
        for finger in joint_definitions[hand]:
            joint_defs = joint_definitions[hand][finger]

            for joint_def in joint_defs:
                joint_name = joint_def['type']
                proximal_id = joint_def['proximal']
                distal_id = joint_def['distal']
                joint_dof = joint_def['dof']

                #if both detected
                if(hand_data[hand][finger][proximal_id]['detected'] and hand_data[hand][finger][distal_id]['detected']):

                    # get rotation matrices
                    R1 = hand_data[hand][finger][proximal_id]['rotation_matrix']
                    R2 = hand_data[hand][finger][distal_id]['rotation_matrix']

                    if R1 is not None and R2 is not None:
                        angles = calculate_joint_angles(R1, R2, joint_dof=joint_dof, method=method, flexion_axis=flexion_axis, abduction_axis=abduction_axis)
                        joint_angles[hand][finger][joint_name] = angles
                    else:
                        joint_angles[hand][finger][joint_name] = None if joint_dof == 1 else {
                            'flexion': None, 'abduction': None
                        }
                
                else:
                        if joint_dof == 1:
                            joint_angles[hand][finger][joint_name] = None
                        else:
                            joint_angles[hand][finger][joint_name] = {
                                'flexion': None,
                                'abduction': None
                            }


def update_marker_data(marker_id, rvec, tvec, rotation_matrix, euler):
    for hand in joint_definitions:
        for finger in joint_definitions[hand]:
            finger_markers = set()
            for joint_def in joint_definitions[hand][finger]:
                finger_markers.add(joint_def['proximal'])
                finger_markers.add(joint_def['distal'])

            if marker_id in finger_markers:
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
    y_offset = 30
    for hand in joint_angles:
        for finger in joint_angles[hand]:
            for joint_name, angle_data in joint_angles[hand][finger].items():
                if angle_data is not None:
                    if isinstance(angle_data, dict):
                        # multi-DOF joint
                        flexion = angle_data.get('flexion')
                        abduction = angle_data.get('abduction')
                        
                        if flexion is not None:
                            text = f"{hand}_{finger}_{joint_name}_flex: {flexion:.1f}°"
                            cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            y_offset += 20
                        
                        if abduction is not None:
                            text = f"{hand}_{finger}_{joint_name}_abd: {abduction:.1f}°"
                            cv.putText(frame, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            y_offset += 20
                    else:
                        # single DOF joint
                        text = f"{hand}_{finger}_{joint_name}: {angle_data:.1f}°"
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