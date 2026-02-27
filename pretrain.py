import argparse
import logging
import os
from itertools import cycle

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from analysis import *
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from partial_fc import PartialFC, PartialFCAdamW

from lr_scheduler import build_scheduler
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

import torch

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )



def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # Recognition dataloader
    # train_loader = get_dataloader(
    #     cfg.rec,
    #     args.local_rank,
    #     cfg.recognition_bz,
    #     cfg.dali,
    #     cfg.seed,
    #     cfg.num_workers
    # )

    train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)
    # Backbone
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph() # Commented out as it might not be needed without DDP

    # 设置梯度累积步数，使其等效于512的批量大小
    gradient_accumulation_steps = 512 // (cfg.recognition_bz * world_size)
    cfg.total_recognition_bz = cfg.recognition_bz * world_size * gradient_accumulation_steps
    cfg.warmup_step = cfg.num_image // cfg.total_recognition_bz * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_recognition_bz * cfg.num_epoch

    # 只保留 recognition 的 batch size
    cfg.total_batch_size = world_size * cfg.recognition_bz * gradient_accumulation_steps

    # 学习率按单任务批量大小进行缩放
    # cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    # cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    # cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0
    # Recognition loss
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])  # only load backbone!
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_epoch_{cfg.resume_epoch}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])

        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.recognition_bz,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()

    amp = torch.cuda.amp.GradScaler(init_scale=1024, growth_interval=2000)

    bzs = [cfg.recognition_bz]

    features_cut = [0 for i in range(2)]
    for i in range(1, 2):
        features_cut[i] = features_cut[i - 1] + bzs[i - 1]


    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, recognition_label) in enumerate(train_loader):
            global_step += 1

            img = img.cuda(non_blocking=True)
            recognition_label = recognition_label.cuda(non_blocking=True)
            local_features, global_features, x = backbone.module.forward_features(img)

            recognition_features = x[features_cut[0]: features_cut[1]]

            local_embeddings = backbone.module.feature(recognition_features)

            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)

            loss = recognition_loss

            if cfg.fp16:
                amp.scale(loss).backward()
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step_update(global_step - 1)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                callback_logging(global_step, loss_am, recognition_loss_am, epoch, cfg.fp16,
                                 opt.param_groups[0]['lr'], amp)
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_epoch_{epoch}_gpu_{rank}.pt"))

        if rank == 0: # rank is 0 for single GPU 
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    with torch.no_grad():
        callback_verification(global_step, backbone)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group() # Commented out for single GPU


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
