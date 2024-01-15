# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
# Import the packages needed.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
# from utils import label_map_util
# from utils import visualization_utils as viz_utils
import six
from six.moves import range
from six.moves import zip


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='basketball_dribble', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

MAX_BOXES_TO_DRAW = 200
MIN_SCORE_THRESH = .4

model = tf.saved_model.load('C:/Users/putte/Downloads/new_model11/content/inference_graph/saved_model')

wc_width = 640
wc_height = 480


def Detect(frame):
    BasketBoxes = []
    frame = np.asarray(frame)
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)    # Display the resulting frame
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > MIN_SCORE_THRESH:
            box = tuple(boxes[i].tolist())
            if (classes[i] == 1):
                BasketBoxes.append((round(100*scores[i]), box))
    BasketBoxes.sort(reverse=True)
    if (len(BasketBoxes) > 0):
        return True, BasketBoxes[0][1]
        ymin, xmin, ymax, xmax = BasketBoxes[0][1]
    else:
        return False, []


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

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    Detected = False
    StateInited = False
    box = []
    f = 0
    init_mod = 10
    mod = init_mod
    for f, im in enumerate(ims):
        if (f % mod == 0):
            # f = 0
            print("finding.......")
            Detected, box = Detect(im)
            StateInited = Detected
            if (not Detected):
                mod += 1
                if (mod > 80) :
                    mod = init_mod
            else:
                f = 0
                mod = init_mod
        if (Detected):
            ymin, xmin, ymax, xmax = box
            x, y, w, h = xmin*wc_width, ymin*wc_height, (xmax-xmin)*wc_width, (ymax-ymin)*wc_height
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            Detected = False
            # print(state)
            # print("------------------")
        if (StateInited):
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]  # draw mask
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 4 points
        cv2.imshow('SiamMask', im)
        f += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
