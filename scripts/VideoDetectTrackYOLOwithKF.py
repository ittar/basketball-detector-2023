import cv2
from KalmanFilter import KalmanFilter
from ultralytics import YOLO
import os
import numpy as np


MODEL_PATH = 'best27.pt'
MODEL_PATH = os.path.join(script_dir, '..', "yolo_models", MODEL_PATH)
model = YOLO(MODEL_PATH)

VID_PATH = ''
VID_PATH = os.path.join("video", VID_PATH)

# using OBS
# VID_PATH = 1

init_mod = 20
mod = init_mod
threshold = .2
f = 0
ModelDetected = False
width = 0
height = 0
pred_box = []

def Detect(frame):
    BasketBoxes = []
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if (score > threshold) :
            if (class_id == 0) :
                BasketBoxes.append((round(100*score), [int(x1),int(y1),int(x2),int(y2)]))
    BasketBoxes.sort(reverse=True)
    if (len(BasketBoxes) > 0):
        return True,BasketBoxes[0][1]
    else:
        return False,[]


if __name__ == "__main__":
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture(VID_PATH)

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    KF = KalmanFilter(0.33, 1, 1, 1, 0.1,0.1)
    x = KF.update(np.array([[1], [1]]))
    print(f'fistdebug = {x}')
    print(f'fistPred = {KF.predict()}')
    while(True):
        # Read frame
        ret, frame = VideoCap.read()
        IsUpdate = False
        # Detect object
        if (f % 4 == 0):
            box = Detect(frame)[1]
            mod = init_mod
            if (len(box) > 0):
                ModelDetected = True
                cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
                width = box[2]-box[0]
                height = box[3]-box[1]
                centers = np.array([[cx], [cy]])
                # Draw the detected circle
                cv2.circle(frame, (int(cx), int(cy)), int(max(width/2,height/2)), (0, 191, 255), 2)
                (x, y) = KF.predict()
                cv2.rectangle(frame, (int(x - width/2), int(y - height/2)), (int(x + width/2), int(y + height/2)), (191, 255, 0), 2)
                (x1, y1) = KF.update(centers)
                # Draw a rectangle as the estimated object position
                cv2.rectangle(frame, (int(x1 - width/2), int(y1 - height/2)), (int(x1 + width/2), int(y1 + height/2)), (0, 0, 255), 2)
            else :
                ModelDetected = False
        else:
            if (ModelDetected and f != 1) :
                (x, y) = KF.predict()
                cv2.rectangle(frame, (int(x - width/2), int(y - height/2)), (int(x + width/2), int(y + height/2)), (255, 0, 0), 2)
        cv2.imshow('image', frame)
        f += 1
        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)