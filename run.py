from pose.dataset import *
from pose.net import *
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from config import config
import argparse
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to the config file (JSON or YAML)')
parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint folder to resume from')
args = parser.parse_args()

# Load configuration from file if provided
if args.config:
    if args.config.endswith('.json'):
        config.from_json(args.config)
    elif args.config.endswith(('.yml', '.yaml')):
        config.from_yaml(args.config)
    else:
        raise ValueError("Config file must be a JSON or YAML file.")
    
accelerator_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

def train():
    if config.use_cuda and torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision=config.amp, gradient_accumulation_steps=config.accumulate, kwargs_handlers=[accelerator_kwargs], log_with="trackio")
        accelerator.init_trackers("pose estimation", config=vars(config))
    else:
        accelerator = Accelerator(cpu=True)
        config.amp = 'no'  # Disable mixed precision if using CPU

    if accelerator.is_main_process:
        config.save()
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Starting training with {config.amp} mixed precision.")
    else:
        logging.basicConfig(level=logging.CRITICAL)

    set_all_seeds(config)
    logger.info(f"Random seed value: {config.seed}")

    os.makedirs(config.weight_dir, exist_ok=True)
    logger.info(f"Weight saved at: {config.weight_dir}")

    model = MyNet(config)
    logger.info("Net created successfully.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    if config.use_pretrained and args.resume is None:
        weight_path = os.path.join(config.weight_dir, config.weight_name)
        if os.path.exists(weight_path):
            logger.info(f"Loading pretrained weights from {weight_path}")
            model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
        else:
            logger.warning(f"Pretrained weights not found at {weight_path}")

    optimizer = build_optimizer(model, config)
    logger.info(f"Optimizer: {config.optimizer}, learning rate: {float(config.learning_rate):.4e}")

    trainloader = build_dataloader(config, is_train=True)
    logger.info(f"Train dataloader created with {len(trainloader)} batches.")

    scheduler = build_scheduler(optimizer, config, len(trainloader))
    logger.info(f"{config.scheduler} scheduler created with {config.max_epochs} epochs.")

    model, optimizer, trainloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, scheduler
    )

    criterion = PoseLoss(config)
    model.train()

    start_epoch = 0
    if args.resume:
        if os.path.isdir(args.resume):
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            accelerator.load_state(args.resume)
            try:
                folder_name = os.path.basename(os.path.normpath(args.resume))
                start_epoch = int(folder_name.split('_')[-1])
                logger.info(f"Resumed successfully. Starting from Epoch {start_epoch + 1}")
            except ValueError:
                logger.warning("Could not parse epoch from checkpoint name. Starting from Epoch 1.")
        else:
            logger.error(f"Checkpoint folder {args.resume} not found!")

    for epoch in range(start_epoch, config.max_epochs):
        train_one_epoch(
            model, trainloader, optimizer, scheduler, criterion, accelerator, epoch, config
        )
        # accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
                unwarp_model = accelerator.unwrap_model(model)
                checkpoint_dir = os.path.join(config.weight_dir, f"checkpoint_epoch_{epoch+1}")
                accelerator.save_state(checkpoint_dir)
                logger.info(f"Checkpoint saved at {checkpoint_dir}")

    if accelerator.is_main_process:
        unwarp_model = accelerator.unwrap_model(model)
        weight_path = os.path.join(config.weight_dir, config.weight_name)
        torch.save(unwarp_model.state_dict(), weight_path)
        logger.info(f"Model weights saved at {weight_path}")

    if accelerator.is_main_process:
        logger.info("Training completed successfully.")

    model.to('cpu')
    accelerator.end_training()
    del model, optimizer, trainloader, scheduler, criterion
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == '__main__':
    train()
