import os
from utils import util, trainer
from utils.tools import MyArgumentParser
from omegaconf import OmegaConf

if __name__ == "__main__":
    ArgParser = MyArgumentParser()
    parser = ArgParser.get_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load("config/voc.yaml")
    util.setup_seed(100)
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, cfg.project_name)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    gpu_ids = ''
    for gpu_id in cfg.train.gpu_ids:
        gpu_ids += str(gpu_id) + ','
    gpu_ids = gpu_ids[:-1]

    model = trainer.MixTrTrainer(args=cfg)
    model.train_model()
