#!/usr/bin/env python
"""Supervised training of ChessNet on shards from data.py.

Keeps the packed dataset (~100 bytes/position) in RAM and expands each batch
to planes on the GPU. Logs to stdout and <out>/log.csv; saves <out>/last.pt
(resumable) and <out>/best.pt (lowest val policy loss).

Usage:
    python train.py --data data/2026-09 --out runs/r1 --epochs 3
    python train.py --data data/2026-09 --out runs/r1 --epochs 3 --resume
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import ChessNet, to_planes


def load_split(directory: str) -> dict[str, torch.Tensor]:
    arrays = data.load_shards(directory)
    arrays["boards"] = arrays["boards"].view(np.int64)        # same bits, torch-friendly dtype
    arrays["moves"] = arrays["moves"].astype(np.int64)
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


def get_batch(split, idx, device):
    batch = {k: v[idx].to(device, non_blocking=True) for k, v in split.items()}
    return to_planes(batch["boards"], batch["meta"]), batch["moves"], batch["values"].float()


def losses(net, x, moves, values, value_weight):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        policy, value = net(x)
    policy, value = policy.float(), value.float()
    p_loss = F.cross_entropy(policy, moves)
    v_loss = F.mse_loss(value, values)
    return p_loss + value_weight * v_loss, p_loss, v_loss, policy


@torch.no_grad()
def validate(net, val, device, batch_size, limit):
    net.eval()
    n = min(limit, len(val["moves"]))
    totals = np.zeros(4)
    for i in range(0, n, batch_size):
        idx = torch.arange(i, min(i + batch_size, n))
        x, moves, values = get_batch(val, idx, device)
        _, p_loss, v_loss, policy = losses(net, x, moves, values, 0.0)
        top5 = policy.topk(5, 1).indices
        totals += len(idx) * np.array([p_loss.item(), v_loss.item(),
                                       (top5[:, 0] == moves).float().mean().item(),
                                       (top5 == moves[:, None]).any(1).float().mean().item()])
    net.train()
    return dict(zip(["val_policy", "val_value_mse", "top1", "top5"], totals / n))


def save(path, **state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dir with train/ and val/ from data.py")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-3, help="peak LR (one-cycle)")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=2000, help="steps between val/checkpoint")
    p.add_argument("--val-limit", type=int, default=200_000, help="val positions per eval")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    train, val = load_split(os.path.join(args.data, "train")), load_split(os.path.join(args.data, "val"))
    n_train = len(train["moves"])
    steps_per_epoch = n_train // args.batch
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"train {n_train:,}  val {len(val['moves']):,}  steps {total_steps:,}  device {device}", flush=True)

    net = ChessNet(args.blocks, args.channels).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.03)
    step, best = 0, float("inf")
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        if ckpt["sched"]["total_steps"] != total_steps:
            raise ValueError(f"checkpoint was for {ckpt['sched']['total_steps']:,} steps, now {total_steps:,}; "
                             "resume with the original --epochs/--batch or start a new --out")
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed at step {step:,}", flush=True)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path) or not args.resume
    log = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "epoch", "lr", "train_loss", "train_policy", "train_value",
                         "val_policy", "val_value_mse", "top1", "top5", "pos_per_s"])

    net.train()
    running, t0, seen = np.zeros(3), time.time(), 0
    while step < total_steps:
        # Permutation is derived from the epoch so a resumed run sees the same order.
        epoch = step // steps_per_epoch
        gen = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(n_train, generator=gen)
        for b in range(step % steps_per_epoch, steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x, moves, values = get_batch(train, idx, device)
            loss, p_loss, v_loss, _ = losses(net, x.contiguous(memory_format=torch.channels_last),
                                             moves, values, args.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += [loss.item(), p_loss.item(), v_loss.item()]
            seen += len(idx)

            if step % args.eval_every == 0 or step == total_steps:
                metrics = validate(net, val, device, args.batch, args.val_limit)
                rate = seen / (time.time() - t0)
                train_avg = running / (seen / args.batch)
                row = [step, round(step / steps_per_epoch, 3), f"{sched.get_last_lr()[0]:.2e}",
                       *np.round(train_avg, 4), *(round(v, 4) for v in metrics.values()), round(rate)]
                writer.writerow(row)
                log.flush()
                print(f"step {step:,}/{total_steps:,}  ep {row[1]}  loss {train_avg[0]:.3f}  "
                      f"val_pol {metrics['val_policy']:.3f}  val_v {metrics['val_value_mse']:.3f}  "
                      f"top1 {metrics['top1']:.1%}  top5 {metrics['top5']:.1%}  {rate:,.0f} pos/s",
                      flush=True)
                state = {"model": net.state_dict(), "config": net.config, "step": step}
                if metrics["val_policy"] < best:
                    best = metrics["val_policy"]
                    save(best_path, **state, metrics=metrics)
                save(last_path, **state, opt=opt.state_dict(), sched=sched.state_dict(), best=best)
                running, t0, seen = np.zeros(3), time.time(), 0
    log.close()


if __name__ == "__main__":
    main()
