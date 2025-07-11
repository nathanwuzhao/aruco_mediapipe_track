import cv2 as cv
import os

#get from webcam
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('could not open camera 0')
    exit()

#doesnt wokr without this line for some reason
cv.namedWindow('test')

#counter for images taken
img_counter = 0
path = "C:/Users/mcsft/OneDrive/Documents/umkc_opencv/calibration_images"


while True:
    #read from camera
    ret, frame = capture.read()
    if not ret:
        print('cannot receive frame, exiting...')
        break
    cv.imshow('test', frame)

    #detect keypresses
    k = cv.waitKey(1)
    if k == ord('q'):
        #q pressed
        print('closing...')
        break
    elif k == ord('f'):
        #f pressed
        img_name = "opencvframe{}.jpg".format(img_counter)
        cv.imwrite(os.path.join(path, img_name), frame)
        print("written to {}!".format(img_name))
        img_counter += 1

#close everything
capture.release()
cv.destroyAllWindows()