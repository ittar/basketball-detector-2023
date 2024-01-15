import os

from ultralytics import YOLO
import cv2

#Use test01 env !!

wc_width = 640
wc_height = 480

MODEL_PATH = 'best19.pt'
MODEL_PATH = os.path.join("yolo_models", MODEL_PATH)

# Load a model
model = YOLO(MODEL_PATH)

threshold = 0.7
VID_PATH = ''
VID_PATH = os.path.join("video", VID_PATH)

class_name_dict = {0: 'basketball', 1: "Hoop",2 : "Hoop"}
wc_width = 640  
wc_height = 480

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wc_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wc_width)

while cap:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (wc_width, wc_height), interpolation=cv2.INTER_LINEAR)
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            print(int(x1), int(y1),int(x2), int(y2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper() + " " + str(int(score*100)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # esc key
        break


cap.release()
cv2.destroyAllWindows()
