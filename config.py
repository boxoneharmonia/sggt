import logging
import json
import yaml
import os

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # Network configuration
        self.base = '/home/box/git/dinov2-base'
        self.num_patches = 576
        self.patch_size = 14
        self.embed_dim = 768
        self.encoder_depth = 12 
        self.decoder_depth = 4
        self.num_heads = 16
        self.groups = 1
        self.mlp_ratio = 4.0 
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_ratio = 0.0
        self.attn_drop_ratio = 0.0
        self.drop_path_ratio = 0.0
        self.dpt_features = 256
        self.feature_idx = [1,3,4,5]
        self.maps = 8

        # Dataset configuration
        self.root = '../SwissCube/'
        self.training = 'training_sequences.json'
        self.testing = 'testing_sequences.json'
        self.ptsfile = 'swisscube_bbox.json'
        self.seq_len = 8
        self.batch_size = 1
        self.num_workers = 6
        self.original_size = [336,336]
        self.scale = 2.0

        # Optimizer configuration
        self.optimizer = 'adamw'
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.momentum = 0.99
        self.z_beta = 0.1
        self.z_weight = 2.0
        self.R_beta = 1.0
        self.R_weight = 0.5
        self.mask_weight = 1.0
        self.cam_weight = 1.0
        self.pts_weight = 1.0
        self.var_weight = 0.1
        self.pcloud_alpha = 0.2
        self.pcloud_weight = 1.0
        self.pconf_weight = 1.0

        # Scheduler configuration
        self.scheduler = 'cosine' 
        # huggingface scheduler: "linear", "cosine", "cosine_with_restarts",
        # "cosine_with_min_lr", "polynomial", "constant", 
        self.max_epochs = 1500
        self.warmup_proportion = 0.05
        self.num_cycles = 0.5 # Number of cycles for cosine/cosine_with_min_lr scheduler (float) or for cosine_with_restarts scheduler (int), must be careful with this parameter
        self.power = 1.0 # Polynomial decay power for polynomial scheduler
        self.min_lr_rate = 1e-8 # Minimum learning rate for cosine_with_min_lr scheduler (not a ratio)

        # Training configuration
        self.max_grad_norm = 1.0
        self.train_log_dir = './log/train'
        self.valid_log_dir = './log/valid'
        self.freeze = False

        self.use_cuda = True
        self.amp = 'bf16' # accelerator mixed precision: 'no', 'fp16', 'bf16'
        self.accumulate = 3
        self.seed = 19260817
        self.weight_dir = './weights'
        self.weight_name = 'weights.pth.tar'
        self.use_pretrained = False
        self.save_interval = 100  # Save model every n epochs
        
    def update(self, config_dict):
        if not isinstance(config_dict, dict):
            logger.error("Argument to update must be a dictionary.")
            return
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # logger.info(f"set {key} as {value}.")
            else:
                logger.warning(f"'{key}' not exist!")

    def from_json(self, json_file):
        """
        Update configuration from a JSON file.
        """
        try:
            with open(json_file, 'r') as f:
                config_dict = json.load(f)
            self.update(config_dict)
        except FileNotFoundError:
            logger.error(f"JSON file not found at: {json_file}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {json_file}")

    def from_yaml(self, yaml_file):
        """
        Update configuration from a YAML file.
        """
        try:
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.update(config_dict)
        except FileNotFoundError:
            logger.error(f"YAML file not found at: {yaml_file}")
        except yaml.YAMLError:
            logger.error(f"Error parsing YAML file: {yaml_file}")
    def save(self, file_path='./log/config.bak.yaml'):
        """
        Save the current configuration to a YAML file.
        """
        dir_name = os.path.dirname(file_path)
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)

        config_dict = {key: value for key, value in self.__dict__.items() 
                       if not key.startswith('__') and not callable(value)}

        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, indent=4, sort_keys=False)
            logger.info(f"Configuration successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}. Error: {e}")

config = Config()

