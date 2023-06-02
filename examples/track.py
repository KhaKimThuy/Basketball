from pathlib import Path
import torch
import argparse
import cv2

from boxmot.tracker_zoo import create_tracker
from detect import Detector, TASK_MAP

from ultralytics.yolo.utils import LOGGER, colorstr, ops, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.data.utils import VID_FORMATS

from multi_yolo_backend import MultiYolo
from utils import write_MOT_results

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # repo root absolute path
EXAMPLES = FILE.parents[0]  # examples absolute path
WEIGHTS = EXAMPLES / 'weights'


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        ROOT /\
        'boxmot' /\
        opt.tracking_method /\
        'configs' /\
        (opt.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half
        )
        predictor.trackers.append(tracker)
        print(f'Trackerrr: {tracker}')


@torch.no_grad()
def run(args):
    # model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    model = Detector(args['yolo_model'])
    overrides = model.overrides.copy()

    # model.task = detection (setup when trainning weight)
    # <class 'ultralytics.yolo.v8.detect.predict.DetectionPredictor'>
    ## model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)
    model.predictor = TASK_MAP['detect'][3](overrides=overrides, _callbacks=model.callbacks)

    # extract task predictor
    # Start with predictor = None
    predictor = model.predictor 

    # combine default predictor args with custom, preferring custom
    ''' COMBINED_ARGS
        {'task': 'detect', 'mode': 'train', 'model': 'D:\\BasketBall\\v8_L.pt', 'data': '/kaggle/input/ball-yaml/ball.yaml', 
        'epochs': 100, 'patience': 50, 'batch': 16, 'imgsz': [640], 'save': False, 'save_period': -1, 'cache': False, 'device': '', 
        'workers': 8, 'project': WindowsPath('D:/BasketBall/Basketball/runs/track'), 'name': 'exp', 'exist_ok': False, 'pretrained': False, 
        'optimizer': 'SGD', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 
        'close_mosaic': 0, 'resume': False, 'amp': True, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 
        'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': 0.5, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 
        'plots': True, 'source': 'D:\\BasketBall\\vids\\Video_2.mp4', 'show': True, 'save_txt': False, 'save_conf': False, 
        'save_crop': False, 'show_labels': False, 'show_conf': False, 'vid_stride': 1, 'line_width': 1, 'visualize': False, 
        'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'boxes': True, 'format': 'torchscript', 
        'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': False, 'opset': None, 'workspace': 4, 
        'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 
        'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 
        'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'cfg': None, 'v5loader': False, 
        'tracker': 'botsort.yaml', 'yolo_model': WindowsPath('D:/BasketBall/v8_L.pt'), 
        'reid_model': WindowsPath('D:/BasketBall/Basketball/osnet_x1_0_msmt17.pt'), 'tracking_method': 'deepocsort'}
    '''
    combined_args = {**predictor.args.__dict__, **args}

    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)
    
    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name, exist_ok=predictor.args.exist_ok)
    
    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')
    model = MultiYolo(
        # model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        model=model.predictor.model,
        device=predictor.device,
        args=predictor.args
    )

    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        # visualize = increment_path(save_dir / Path(path[0]).stem, exist_ok=True, mkdir=True) if predictor.args.visualize and (not predictor.dataset.source_type.tensor) else False

        n = len(im0s)
        predictor.results = [None] * n
        
        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model(im, im0s)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')
        
        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            
            if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)
            
            # write inference results to a file or directory   
            if predictor.args.verbose or predictor.args.save or predictor.args.save_txt or predictor.args.show:
                s += predictor.write_results(i, predictor.results, (p, im, im0))

                # plot_args = dict(line_width=self.args.line_width,
                #                 boxes=self.args.boxes,
                #                 conf=self.args.show_conf,
                #                 labels=self.args.show_labels)

                predictor.txt_path = Path(predictor.txt_path)
                
                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                else:
                    # append folder name containing current img
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name
                    
                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_txt:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        predictor.run_callbacks('on_predict_batch_end')

        # print time (inference-only)
        if predictor.args.verbose:
            LOGGER.info(f'{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
                    f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=1, help='line_width')
    parser.add_argument('--show-labels', type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument('--show-conf', type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    # parser.add_argument('--hide-label', action='store_true', help='hide labels when show')
    # parser.add_argument('--hide-conf', action='store_true', help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)