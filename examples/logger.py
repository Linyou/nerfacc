from loguru import logger
import os
import cv2
import torch
import imageio
import numpy as np

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

results_root = '/home/loyot/workspace/code/training_results/nerfacc'
class Logger(object):
    def __init__(self, args, log_dir=results_root):
        self.log_dir = log_dir
        self.log_file = None
        
        feat_dir = ''

        if args.use_feat_predict:
            feat_dir += 'pf' 

        if args.use_weight_predict:
            feat_dir += '_pw'

        if args.distortion_loss:
            feat_dir += "_distor"

        if args.use_time_embedding:
            feat_dir += "_te"
            if args.use_time_attenuation:
                feat_dir += "_ta"

        if args.weight_rgbper:
            feat_dir += "_rgbper"

        if args.acc_entorpy_loss:
            feat_dir += "_acc"

        self.feat_dir = feat_dir
        str_lr = str(args.lr).replace('.', '-')
        self.image_root = f'{log_dir}/dngp/{args.scene}/lr_{str_lr}/steps_{args.max_steps}/{feat_dir}/'
        self.args = args

        logger.remove(0)
        logger.add(os.path.join(self.image_root, 'logs', "{time}.log"))

    def save_state(self, radi, occ, step):
        str_lr = str(self.args.lr).replace('.', '-')
        os.makedirs(f'{self.log_dir}/checkpoints', exist_ok=True)
        torch.save(
            {
                "radiance_field": radi.state_dict(),
                "occupancy_grid": occ.state_dict(),
            },
            f"{self.log_dir}/checkpoints/dngp_lr_{str_lr}_{self.feat_dir}_{self.args.scene}_{step}.pth",
        )

    def save_image(self, rgb, depth, step=0, pixels=None):

        os.makedirs(f'{self.image_root}', exist_ok=True)
        assert os.path.exists(f'{self.image_root}'), f"test images saving path dose not exits! path: {self.image_root}"
        imageio.imwrite(
            f"{self.image_root}/depth_{step}.png",
            depth2img(depth),
        )
        imageio.imwrite(
            f"{self.image_root}/rgb_{step}.png",
            (rgb.cpu().numpy() * 255).astype(np.uint8),
        )
        if pixels is not None:
            imageio.imwrite(
                f"{self.image_root}/rgb_error_{step}.png",
                (
                    (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                ).astype(np.uint8),
            )

    def log(self, msg):
        logger.info(msg)
