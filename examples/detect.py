# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path
from typing import Union

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, SegmentationModel,
                                  attempt_load_one_weight, guess_model_task, nn, yaml_model_load)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import (LOGGER, ROOT, callbacks, is_git_dir)
from ultralytics.yolo.utils.checks import check_file
from ultralytics.yolo.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
        yolo.v8.classify.ClassificationPredictor],
    'detect': [
        DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
        yolo.v8.detect.DetectionPredictor],
    'segment': [
        SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
        yolo.v8.segment.SegmentationPredictor],
    'pose': [PoseModel, yolo.v8.pose.PoseTrainer, yolo.v8.pose.PoseValidator, yolo.v8.pose.PosePredictor]
    }


class Detector:
    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            from ultralytics.hub.session import HUBTrainingSession
            self.session = HUBTrainingSession(model)
            model = self.session.model_file

        # Load or create new YOLO model
        suffix = Path(model).suffix
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt
        if suffix == '.yaml':
            self._new(model, task)
        else:
            self._load(model, task)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model."""
        return any((
            model.startswith('https://hub.ultra'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID


    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str) or (None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    @smart_inference_mode()
    def load(self, weights='yolov8n.pt'):
        """
        Transfers parameters with matching names and shapes from 'weights' to model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The tracking results.

        """
        if not hasattr(self.predictor, 'trackers'):
            from ultralytics.tracker import register_tracker
            register_tracker(self, persist)
        # ByteTrack-based method needs low confidence predictions as input
        conf = kwargs.get('conf') or 0.1
        kwargs['conf'] = conf
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self._check_is_pytorch_model()
        self.model.to(device)



    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None


    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)


    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
