import os
import sys
import time
import torch
from loguru import logger


class Config:
    def __init__(self):
        # global
        self.seed = 0
        self.cuDNN = True
        self.debug = False
        self.num_workers = 0
        self.str_time = time.strftime("%Y-%m-%dT%H%M%S", time.localtime(time.time()))
        # path
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, "data")
        self.in_path = os.path.join(self.dataset_dir, "fixed_couplets_in.txt")
        self.out_path = os.path.join(self.dataset_dir, "fixed_couplets_out.txt")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.save_dir = os.path.join(self.log_dir, self.str_time)
        self.img_save_dir = os.path.join(self.save_dir, "images")
        self.model_save_dir = os.path.join(self.save_dir, "checkpoints")
        for path in (
            self.log_dir,
            self.save_dir,
            self.img_save_dir,
            self.model_save_dir,
        ):
            if not os.path.exists(path):
                os.makedirs(path)
        # model
        self.d_model = 256
        self.num_head = 8
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 1024
        self.dropout = 0.1
        # train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.val_ratio = 0.1
        self.epochs = 20
        self.warmup_ratio = 0.12
        self.lr_max = 1e-3
        self.lr_min = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.weight_decay = 0.01
        self.early_stop = True
        self.patience = 4
        self.delta = 0
        # log
        logger.remove()
        level_std = "DEBUG" if self.debug else "INFO"
        logger.add(
            sys.stdout,
            colorize=True,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green>|<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>] >>> <level>{message}</level>",
            level=level_std,
        )
        logger.add(
            os.path.join(self.save_dir, f"{self.str_time}.log"),
            format="[{time:YYYY-MM-DD HH:mm:ss,SSS}|{level: <8}|{name}:{function}:{line}] >>> {message}",
            level="INFO",
        )
        logger.info("### Config:")
        for key, value in self.__dict__.items():
            logger.info(f"### {key:20} = {value}")
