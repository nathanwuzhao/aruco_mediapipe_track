import cv2 as cv
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv.namedWindow(previewName)
    cam = cv.VideoCapture(camID)
    if cam.isOpened():
        ret, frame = cam.read()
    else:
        ret = False

    while ret:
        cv.imshow(previewName, frame)
        ret, frame = cam.read()
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyWindow(previewName)

thread0 = camThread('camera 0', 0)
thread0.start()
thread0 = camThread('camera 1', 1)
thread0.start()

