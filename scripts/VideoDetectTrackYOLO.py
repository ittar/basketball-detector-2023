# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from tools.test import *
import numpy as np
import cv2
import os
from ultralytics import YOLO
import tensorflow as tf

print(cv2.__version__)

script_dir = os.path.dirname(os.path.realpath(__file__))
siam_mask_model_path = os.path.join(script_dir, '..', 'siamask_model', 'SiamMask_DAVIS.pth')

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default=siam_mask_model_path, type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='data/basketball2', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

MODEL_PATH = 'best17.pt'
MODEL_PATH = os.path.join(script_dir, '..', "yolo_models", MODEL_PATH)

# Load a model
model = YOLO(MODEL_PATH)  # load a custom model
VID_PATH = '5 Shooting Drills to Become a Better Basketball Player l Individual Shooting Workout_Trim_1.mp4'
VID_PATH = os.path.join("video", VID_PATH)
VID_PATH = 1
SCORE_FRAME = 5
threshold = .2
offset = 0

class_name_dict = {0: 'basketball', 1: "Hoop"}

wc_width = 640
wc_height = 480
score = 0
frameScore = 0
Detected = False
StateInited = False
DetectRim = False
Mask = False
CheckDirect = False
box = []
rim = []
f = 0
cframe = 0
init_mod = 10
mod = init_mod
prev_ball = []
direct_ball = []
scored_this_shooting = False

def Detect(frame):
    global Detected,box,StateInited,rim,DetectRim
    BasketBoxes = []
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if (score > threshold) :
            if (class_id == 0) :
                BasketBoxes.append((round(100*score), [int(x1),int(y1),int(x2),int(y2)]))
            elif (class_id == 1) :
                DetectRim = True
                rim = [int(x1)-offset,int(y1)-offset,int(x2)+offset,int(y2)+offset]
    BasketBoxes.sort(reverse=True)
    if (len(BasketBoxes) > 0):
        return True,BasketBoxes[0][1]
    else:
        return False,[]

def CheckScore (xmin,ymin,xmax,ymax,cx,cy) :
    global score,frameScore,scored_this_shooting,prev_ball,CheckDirect,cframe
    rmx,rmy,rxx,rxy = rim
    if (ymax >= rmy and ymin <= rmy and ((xmin >= rmx and xmax <= rxx) or (xmax >= rmx and xmax <= rxx))) :
        prev_ball = np.array([cx,cy])
        CheckDirect = True
        scored_this_shooting = False
        cframe = 0
    elif (CheckDirect) :
        if (ymin <= rxy and xmin >= rmx and xmax <= rxx) :
            direct_ball = np.array([cx,cy]) - prev_ball
            if (np.dot(direct_ball,np.array([0,1])) > 0 and not scored_this_shooting) :
                score += 1
                scored_this_shooting = True
            CheckDirect = False
        elif (cframe == SCORE_FRAME) : CheckDirect = False
        cframe += 1

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Select ROI
    cv2.namedWindow("SiamMaskYolo", cv2.WND_PROP_FULLSCREEN)
    cap = cv2.VideoCapture(VID_PATH)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wc_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wc_width)

    while (True):
        ret, im = cap.read()
        im = cv2.resize(im, (wc_width, wc_height), interpolation=cv2.INTER_LINEAR)
        if (f % mod == 0):
            Detected, box = Detect(im)
            StateInited = Detected
            f = 0
            if (not Detected):
                f = -1
            cv2.putText(im, "DETECT!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3, cv2.LINE_AA)
        if (Detected):
            xmin, ymin, xmax, ymax = box
            x, y, w, h = xmin, ymin, (xmax-xmin), (ymax-ymin)
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            init_state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            state = init_state
            Detected = False
            xmin,ymin,xmax,ymax = target_pos[0] - target_sz[0]/2, target_pos[1] - target_sz[1]/2, target_pos[0] + target_sz[0]/2, target_pos[1] + target_sz[1]/2
            center = np.array([int((xmin+xmax)/2),int((ymin+ymax)/2)])               
            if (DetectRim) : CheckScore(xmin,ymin,xmax,ymax,center[0],center[1])
            cv2.circle(im, center, 2, (0, 0, 255), 3)
            cv2.rectangle(im,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0, 0, 255), 4)
        elif (StateInited):
            state = siamese_track(state, im, mask_enable=Mask, refine_enable=True, device=device)  # track
            if (Mask) :
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]  # draw mask
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 4 points
            else :
                target_pos = state['target_pos']
                target_sz = state['target_sz']
                xmin,ymin,xmax,ymax = target_pos[0] - target_sz[0]/2, target_pos[1] - target_sz[1]/2, target_pos[0] + target_sz[0]/2, target_pos[1] + target_sz[1]/2
                center = np.array([int((xmin+xmax)/2),int((ymin+ymax)/2)])                
                if (DetectRim) : 
                    CheckScore(xmin,ymin,xmax,ymax,center[0],center[1])
                cv2.circle(im, center, 2, (0, 0, 255), 3)
                cv2.rectangle(im,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0, 255, 0), 4)
        if (DetectRim) :
            cv2.rectangle(im,(int(rim[0]),int(rim[1])),(int(rim[2]),int(rim[3] - (rim[3]-rim[1])/2)),(0, 0, 255), 3)
            cv2.rectangle(im,(int(rim[0]),int(rim[1])),(int(rim[2]),int(rim[3])),(255, 0, 0), 4)
        cv2.putText(im, str(score), (wc_width-60, wc_height-60),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('SiamMaskYolo', im)
        f += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()
