# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

script_dir = os.path.dirname(os.path.realpath(__file__))
siam_mask_model_path = os.path.join(script_dir, '..', 'siamMaskModel', 'SiamMask_DAVIS.pth')

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default=siam_mask_model_path, type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='data/basketball2', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

# def get_layers(model: torch.nn.Module):
#     children = list(model.children())
#     return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]


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
    # for layer in get_layers(siammask) :
    #     print(layer)

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    ret, im = cap.read()
    try:
        init_rect = cv2.selectROI('SiamMask', im, False, False)
        x, y, w, h = init_rect
    except:
        exit()
    tic = cv2.getTickCount()
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    init_state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
    state = init_state 
    toc = cv2.getTickCount() - tic
    f = 1
    while True:
        tic = cv2.getTickCount()
        ret, im = cap.read()
        state = siamese_track(init_state, im, mask_enable=True, refine_enable=True, device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]  # draw mask
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 4 points
        # print(np.int0(location).reshape((-1, 1, 2)))
        # print("---------")
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key == 27:
            break
        f += 1
        toc += cv2.getTickCount() - tic
        # print(toc / cv2.getTickFrequency())
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
