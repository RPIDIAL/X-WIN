import os
import copy
import argparse
import logging
import time
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

from utils import setup_distributed, cleanup_distributed, is_main_process, all_reduce_loss, setup_paramgroup
from datasets.xray_proj import XrayProjDataset, ProjCollator
from datasets.mimic_cxr import MimicDataset
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from datasets.utils import get_transform, get_val_transform
from src.models.vision_transformer import vit_base, vit_small, vit_predictor
from src.utils.logging import AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
parser.add_argument("--train_txt", type=str, default="/fast/yangz16/outputs/x-win/train_drrs3.txt")
parser.add_argument("--test_txt", type=str, default="/fast/yangz16/outputs/x-win/test_drrs3.txt")
parser.add_argument("--train_real_txt", type=str, default="/fast/yangz16/outputs/x-win/train_mimic.txt")
parser.add_argument("--test_real_txt", type=str, default="/fast/yangz16/outputs/x-win/test_mimic.txt")
# parser.add_argument("--objective", type=str, default="novelview")
parser.add_argument("--size", type=int, default=224)
# parser.add_argument("--crop_scale", type=tuple, default=(0.75, 1.0))
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--bs", type=int, default=24)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--start_lr", type=float, default=2e-5)
parser.add_argument("--final_lr", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--ema", type=tuple, default=(0.996, 1.0))
parser.add_argument("--wd", type=float, default=0.004)
parser.add_argument("--final_wd", type=float, default=0.04)
parser.add_argument("--predictor_depth", type=int, default=3)
parser.add_argument("--loss_type", type=str, default="contrastive", choices=["mse", "contrastive"])
parser.add_argument("--contrastive_temp", type=float, default=0.1)
parser.add_argument("--recon_weight", type=float, default=1.0)


def compute_alignment_loss(z, h, loss_type, mse_criterion, temperature=0.1):
    if loss_type == "mse":
        return mse_criterion(z, h)
    if loss_type == "contrastive":
        z = F.normalize(z, dim=1)
        h = F.normalize(h, dim=1)
        logits = torch.matmul(z, h.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_zh = F.cross_entropy(logits, labels)
        loss_hz = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_zh + loss_hz)
    raise ValueError(f"Unknown loss type: {loss_type}")


def sample_masks(mask_collator, batch_size, device):
    _, masks_enc, masks_pred = mask_collator([0] * batch_size)
    masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
    masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
    return masks_enc, masks_pred


def build_mask_targets(target_encoder, imgs, masks_pred, repeat_count):
    h = target_encoder(imgs)
    h = F.layer_norm(h, (h.size(-1),))
    bsz = len(h)
    h = apply_masks(h, masks_pred)
    h = repeat_interleave_batch(h, bsz, repeat=repeat_count)
    return h


def train(args):
    # Set up distributed
    setup_distributed(args.local_rank)

    # Logging arguments
    if is_main_process():
        print(vars(args))

    # Set up device
    device = torch.device("cuda", args.local_rank)
    if args.outdir is not None and is_main_process():
        os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(42)

    # Compute number of patches
    num_patches = (args.size // args.patch_size) ** 2

    # Set up encoder and predictors
    encoder = vit_base().to(device)
    encoder = DDP(encoder, device_ids=[args.local_rank], output_device=args.local_rank)

    predictor_align = vit_predictor(num_patches=num_patches, depth=args.predictor_depth).to(device)
    predictor_align = DDP(predictor_align, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    predictor_recon = vit_predictor(num_patches=num_patches, depth=args.predictor_depth).to(device)
    predictor_recon = DDP(predictor_recon, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    target_encoder = copy.deepcopy(encoder.module).to(device)
    target_encoder = DDP(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Set up datasets and dataloader
    train_transform = get_val_transform(image_size=args.size)
    # train_transform = get_transform(image_size=args.size, crop_scale=args.crop_scale)
    val_transform = get_val_transform(image_size=args.size)
    train_ds_drr = XrayProjDataset(root_dir=args.train_txt, split='train', transform=train_transform, ctx_only=True)
    val_ds_drr = XrayProjDataset(root_dir=args.test_txt, split='val', transform=val_transform, ctx_only=False)
    train_ds_real = MimicDataset(split='train', train_txt=args.train_real_txt, test_txt=args.test_real_txt, transform=train_transform)
    val_ds_real = MimicDataset(split='test', train_txt=args.train_real_txt, test_txt=args.test_real_txt, transform=val_transform)
    proj_collator = ProjCollator(step=15, num_samples=args.num_samples, transform=train_ds_drr.transform)
    mask_collator = MBMaskCollator(
        input_size=args.size,
        patch_size=args.patch_size,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.0),
        nenc=1,
        npred=1,
        allow_overlap=False,
        min_keep=4,
    )

    train_sampler_drr = DistributedSampler(train_ds_drr, shuffle=True)
    val_sampler_drr = DistributedSampler(val_ds_drr, shuffle=False)
    train_sampler_real = DistributedSampler(train_ds_real, shuffle=True)
    val_sampler_real = DistributedSampler(val_ds_real, shuffle=False)
    train_loader_drr = DataLoader(
        train_ds_drr, batch_size=args.bs, collate_fn=proj_collator, sampler=train_sampler_drr,
        num_workers=8, pin_memory=False, persistent_workers=True, drop_last=True
    )
    val_loader_drr = DataLoader(
        val_ds_drr, batch_size=args.bs, sampler=val_sampler_drr,
        num_workers=8, pin_memory=False, drop_last=True
    )
    train_loader_real = DataLoader(
        train_ds_real, batch_size=args.bs, sampler=train_sampler_real,
        num_workers=8, pin_memory=False, persistent_workers=True, drop_last=True
    )
    val_loader_real = DataLoader(
        val_ds_real, batch_size=args.bs, sampler=val_sampler_real,
        num_workers=8, pin_memory=False, drop_last=True
    )

    # Set up optimizer and learning rate scheduler
    ipe = len(train_loader_real)
    param_group = setup_paramgroup(encoder, predictor_align)
    param_group += [
        {
            'params': (p for n, p in predictor_recon.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor_recon.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    optimizer = torch.optim.AdamW(param_group)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(args.warmup*ipe), start_lr=args.start_lr,
                                     ref_lr=args.lr*(args.bs*dist.get_world_size())/256., final_lr=args.final_lr,
                                     T_max=int(args.epochs*ipe))
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=args.wd, final_wd=args.final_wd, T_max=args.epochs*ipe)
    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(ipe*args.epochs) for i in range(int(ipe*args.epochs)+1))
    criterion_mse = nn.MSELoss().to(device)

    # Log loss values
    loss_meter = AverageMeter()
    align_loss_meter = AverageMeter()
    recon_real_loss_meter = AverageMeter()
    recon_drr_loss_meter = AverageMeter()
    dtime_meter, mtime_meter = AverageMeter(), AverageMeter()
    train_loss, val_loss = [], []

    for epoch in range(1, args.epochs+1):
        # Training for one epoch
        encoder.train()
        predictor_align.train()
        predictor_recon.train()
        train_sampler_drr.set_epoch(epoch)
        train_sampler_real.set_epoch(epoch)
        iter_start = time.time()
        for i, (img_real, (batch, img_tgt, action)) in enumerate(zip(train_loader_real, cycle(train_loader_drr))):
            data_end = time.time()

            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()

            model_start = time.time()
            img_ctx = batch[0].to(device, non_blocking=True)
            img_tgt, action = img_tgt.to(device, non_blocking=True), action.to(device, non_blocking=True)
            img_real = img_real.to(device, non_blocking=True)

            # Alignment branch (current MSE/contrastive path)
            z = encoder(img_ctx, masks=None)
            z_view = z.unsqueeze(1).expand(-1, args.num_samples, -1, -1)  # (b, n, l, c)
            z_view = z_view.flatten(0, 1)  # (b*n, l, c)
            z = predictor_align(z_view, action=action)  # (b*n, l, c)
            z = z.mean(dim=1)

            # Alignment targets
            with torch.no_grad():
                h = target_encoder(img_tgt)  # (b*n, l, c)
                h = h.mean(dim=1)

            loss_align = compute_alignment_loss(
                z, h, args.loss_type, criterion_mse, temperature=args.contrastive_temp
            )

            # Reconstruction branch (new masked predictor on DRR + real)
            img_drr = img_ctx
            masks_enc_drr, masks_pred_drr = sample_masks(mask_collator, img_drr.size(0), device)
            masks_enc_real, masks_pred_real = sample_masks(mask_collator, img_real.size(0), device)

            z_drr_rec = encoder(img_drr, masks=masks_enc_drr)
            z_drr_rec = predictor_recon(z_drr_rec, masks_x=masks_enc_drr, masks=masks_pred_drr)
            z_real_rec = encoder(img_real, masks=masks_enc_real)
            z_real_rec = predictor_recon(z_real_rec, masks_x=masks_enc_real, masks=masks_pred_real)

            with torch.no_grad():
                h_drr_rec = build_mask_targets(target_encoder, img_drr, masks_pred_drr, repeat_count=len(masks_enc_drr))
                h_real_rec = build_mask_targets(target_encoder, img_real, masks_pred_real, repeat_count=len(masks_enc_real))

            loss_recon_drr = criterion_mse(z_drr_rec, h_drr_rec)
            loss_recon_real = criterion_mse(z_real_rec, h_real_rec)
            loss_recon = 0.5 * (loss_recon_drr + loss_recon_real)
            loss = loss_align + args.recon_weight * loss_recon
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update encoder
            with torch.no_grad():
                m = next(momentum_scheduler)
                torch._foreach_mul_(list(target_encoder.parameters()), m)
                torch._foreach_add_(list(target_encoder.parameters()), list(encoder.module.parameters()), alpha=1 - m)

            reduced_loss = all_reduce_loss(loss)
            reduced_loss_align = all_reduce_loss(loss_align)
            reduced_loss_recon_real = all_reduce_loss(loss_recon_real)
            reduced_loss_recon_drr = all_reduce_loss(loss_recon_drr)
            iter_end = time.time()
            loss_meter.update(reduced_loss.item(), dist.get_world_size()*len(img_ctx))
            align_loss_meter.update(reduced_loss_align.item(), dist.get_world_size()*len(img_ctx))
            recon_real_loss_meter.update(reduced_loss_recon_real.item(), dist.get_world_size()*len(img_ctx))
            recon_drr_loss_meter.update(reduced_loss_recon_drr.item(), dist.get_world_size()*len(img_ctx))
            mtime_meter.update(iter_end - model_start)
            dtime_meter.update(data_end - iter_start)

            if (i+1) % 10 == 0 and is_main_process():
                print(f"Iter: {i+1}, loss {loss_meter.avg:.6f}, align {align_loss_meter.avg:.6f}, "
                      f"recon_real {recon_real_loss_meter.avg:.6f}, recon_drr {recon_drr_loss_meter.avg:.6f}, "
                      f"model time: {iter_end-model_start:.4f}, "
                      f"lr: {_new_lr:.6f}, wd: {_new_wd:.6f}, teacher momentum: {m:.6f}")
            iter_start = iter_end

        train_loss.append(loss_meter.avg)
        if is_main_process():
            print(f"Train -- Epoch: {epoch}, loss {loss_meter.avg:.6f}, align {align_loss_meter.avg:.6f}, "
                  f"recon_real {recon_real_loss_meter.avg:.6f}, recon_drr {recon_drr_loss_meter.avg:.6f}, "
                  f"data time {dtime_meter.avg:.6f}, model time {mtime_meter.avg:.6f} "
                  f"lr: {_new_lr:.6f}, wd: {_new_wd:.6f}, teacher momentum: {m:.6f}")
        loss_meter.reset(); align_loss_meter.reset(); recon_real_loss_meter.reset(); recon_drr_loss_meter.reset()
        dtime_meter.reset(); mtime_meter.reset()

        #  Validation for one epoch
        with torch.no_grad():
            encoder.eval()
            predictor_align.eval()
            predictor_recon.eval()
            for i, (img_real, (img_ctx, img_tgt, action)) in enumerate(zip(val_loader_real, cycle(val_loader_drr))):
                img_tgt, img_ctx, action = img_tgt.to(device), img_ctx.to(device), action.to(device)
                img_real = img_real.to(device, non_blocking=True)

                # Alignment branch
                z = encoder(img_ctx, masks=None)
                z = predictor_align(z, action=action)
                z = z.mean(dim=1)

                # Alignment targets
                h = target_encoder(img_tgt)
                h = h.mean(dim=1)

                loss_align = compute_alignment_loss(
                    z, h, args.loss_type, criterion_mse, temperature=args.contrastive_temp
                )

                # Reconstruction branch
                img_drr = img_ctx
                masks_enc_drr, masks_pred_drr = sample_masks(mask_collator, img_drr.size(0), device)
                masks_enc_real, masks_pred_real = sample_masks(mask_collator, img_real.size(0), device)

                z_drr_rec = encoder(img_drr, masks=masks_enc_drr)
                z_drr_rec = predictor_recon(z_drr_rec, masks_x=masks_enc_drr, masks=masks_pred_drr)
                z_real_rec = encoder(img_real, masks=masks_enc_real)
                z_real_rec = predictor_recon(z_real_rec, masks_x=masks_enc_real, masks=masks_pred_real)

                h_drr_rec = build_mask_targets(target_encoder, img_drr, masks_pred_drr, repeat_count=len(masks_enc_drr))
                h_real_rec = build_mask_targets(target_encoder, img_real, masks_pred_real, repeat_count=len(masks_enc_real))

                loss_recon_drr = criterion_mse(z_drr_rec, h_drr_rec)
                loss_recon_real = criterion_mse(z_real_rec, h_real_rec)
                loss_recon = 0.5 * (loss_recon_drr + loss_recon_real)
                loss = loss_align + args.recon_weight * loss_recon
                reduced_loss = all_reduce_loss(loss)
                reduced_loss_align = all_reduce_loss(loss_align)
                reduced_loss_recon_real = all_reduce_loss(loss_recon_real)
                reduced_loss_recon_drr = all_reduce_loss(loss_recon_drr)
                loss_meter.update(reduced_loss.item(), dist.get_world_size()*len(img_ctx))
                align_loss_meter.update(reduced_loss_align.item(), dist.get_world_size()*len(img_ctx))
                recon_real_loss_meter.update(reduced_loss_recon_real.item(), dist.get_world_size()*len(img_ctx))
                recon_drr_loss_meter.update(reduced_loss_recon_drr.item(), dist.get_world_size()*len(img_ctx))

            val_loss.append(loss_meter.avg)
            if is_main_process():
                print(f"Val -- Epoch: {epoch}, loss {loss_meter.avg:.6f}, align {align_loss_meter.avg:.6f}, "
                      f"recon_real {recon_real_loss_meter.avg:.6f}, recon_drr {recon_drr_loss_meter.avg:.6f}")
            loss_meter.reset(); align_loss_meter.reset(); recon_real_loss_meter.reset(); recon_drr_loss_meter.reset()

    if is_main_process():
        # Save last checkpoint
        ckpt = {
            'epoch': epoch,
            'encoder': encoder.module.state_dict(),
            'predictor_align': predictor_align.module.state_dict(),
            'predictor_recon': predictor_recon.module.state_dict(),
            'target_encoder': target_encoder.module.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.outdir, 'last_checkpoint.pth'))
        print(f"Saved last_checkpoint.pth for epoch {epoch}")

        # Plot and save loss curves
        x = range(1, args.epochs+1)
        plt.plot(x, train_loss, label='train loss')
        plt.plot(x, val_loss, label='val loss')
        plt.legend(); plt.tight_layout()
        plt.savefig(args.outdir + '/loss_curves.png')

    cleanup_distributed()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
