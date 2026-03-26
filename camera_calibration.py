import cv2 as cv
import numpy as np
import glob 

squares_X = 11 #number of squares on x
squares_Y = 8 #number of squares on y
nX = squares_X - 1 #number of interior corners on X
nY = squares_Y - 1 #number of interniorc corners on Y
square_size = 20 #size in mm of a square side

#termination critera, accuracy or iterations
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.0005) 
 
#3d world coords for checkerboard
object_points3D = np.zeros((nX * nY, 3), np.float32)
object_points3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) * square_size

#store vectors of 3d points and 2d points
object_points = []
image_points = []

#folder where images are stored
images = glob.glob("C:/Users/mcsft/OneDrive/Documents/umkc_opencv/calibration_images/*.jpg")

#go through each image
for image_file in images:

    #load and convert to grayscale
    image = cv.imread(image_file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #find corners
    ret, corners = cv.findChessboardCorners(gray, (nY, nX), None)

    if ret:
        object_points.append(object_points3D)
        corners_subpix = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        image_points.append(corners_subpix)

        #display corners on this jawn
        cv.drawChessboardCorners(image, (nY, nX), corners_subpix, ret)
        cv.imshow("image", image)
        cv.waitKey(2000)

    #perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    #save params
    cv_file = cv.FileStorage('calibration_chessboard.yaml', cv.FILE_STORAGE_WRITE)
    cv_file.write('K', camera_matrix)
    cv_file.write('D', dist_coeffs)
    cv_file.release()

    #print the calibration results
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    cv.destroyAllWindows()