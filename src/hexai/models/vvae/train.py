"""Training loop for the geometric VVAE. run.py starts one, eval.py scores it.

Checkpoints go to .runs/ under the run's wandb name, with its config beside them as a json.
Findings live in .agents/learned.md.

Author: Marco Cassar (@Ocramaru)
"""

import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm as _tqdm

from hexai.data import add_depth_noise, add_pose_noise, environment_of, split_context_target
from hexai.models.vvae.eval import baselines, summarise, validate
from hexai.models.vvae.visuals import (log_context_strip, log_latent, log_novel_view,
                                       log_pose_sweep, save_latent)
from hexai.utilities import PeriodicRestarts, answerable_mask

CHECKPOINTS = Path(__file__).parent / ".runs"  # gitignored, beside the code it belongs to

# a bar redraws over itself on a terminal but floods a redirected log, so drop it when piped
tqdm = partial(_tqdm, file=sys.stdout, ncols=100, mininterval=2.0, disable=not sys.stdout.isatty())

METRIC_NOTES = """\
wandb has no per-metric description, so the glossary lives here.

skill             0 is no better than predicting the val set's mean image, 1 is perfect.
                  The scale for recon, which has none on its own. Printed as skill/skill_
                  answerable, and on a masked run only the second one means anything.
skill_answerable  the same scale over the pixels a context frame actually saw, against a floor
                  measured over those pixels too. A masked run is never taught the rest, so the
                  whole-frame skill charges it for error no gradient ever reached.
recon_answerable  recon over pixels a context frame actually saw, per pixel. The fair
                  comparison when some runs train with the mask and some do not.
selection_score   the smoothed recon that -best.pt is chosen on, so the chart shows what did the
                  choosing. One validation of this swings about as far as a whole run improves,
                  so picking the best raw epoch picks the luckiest draw rather than the best model.
skill_vs_copy     the same again, against handing back the anchor frame unchanged, which is what
                  a system doing no work would answer. Without jitter a window spans ~4 degrees
                  and copying is hard to beat; with it, ~21, and copying loses to the mean image.
unknown_gap       mean predicted observedness where the carving saw, minus where it did not. 1 is
                  perfect separation, 0 is a constant. unknown_mae cannot tell those apart, since
                  most of the box is unobserved and guessing so scores well.
unknown_mae       the same channel in absolute terms. Read it beside the gap, never alone.
lift_fidelity     only with --learned_lift: how much the learned lift's volume resembles the
                  one exact geometry would have produced. Not a target -- a lift that only reproduces the
                  hard one has bought nothing -- but near zero means it is not placing geometry.
lift_agreement    does the decoded volume match the geometry that went in, over observed
                  voxels only. Blind to unobserved space by construction.
invented_density  density the decoder puts where no camera reached. Lower is better, but an
                  empty volume wins it outright, so read it beside lift_agreement.
occluded_density  density sitting where the render's transmittance has already collapsed, which
                  is what --occlusion charges for. A different quantity from invented_density:
                  that one is unseen by the carving, this one is unseen by this query's render,
                  so a voxel a context frame saw still counts here if the query is behind a wall.
                  Logged whether or not the penalty is on, and an empty volume wins it outright.
scene_corr        how alike different scenes decode. Near 1 means the decoder is ignoring
                  its input. A model emitting noise also scores well, so never read alone.
depth_mae_m       depth error in metres, the one number with a physical unit.
decoded_solid     fraction of the decoded volume above the occupancy threshold, against
lifted_occupied   the same fraction of the volume that was lifted from depth. Matching them
                  is the goal; decoded far above lifted is a decoder filling in the room.

Figures are under media/, three scenes each. scene0 and scene1 are fixed val windows, chosen
by index so they are the same in every run; roving is drawn fresh from training each time, so
it shows what the model is being fed rather than the one held-out building.

wandb keeps the running best of the four that have one, under a suffixed key: read
val/lift_agreement.max and val/invented_density.min, not the bare names. The summary a sweep
reads otherwise keeps the last logged value, and these swing between epochs.
"""


Residual = Literal["mse", "l1", "huber"]
DepthSpace = Literal["linear", "log", "inverse"]
Scheduler = Literal["none", "onecycle", "periodic"]

RESIDUALS: dict[Residual, Callable] = {  # what the recon term asks of an uncertain ray, which decides blur
  "mse": lambda e: e.pow(2),  # minimised by the conditional mean, so a bimodal ray renders between
  "l1": lambda e: e.abs(),  # by the median, so it commits to one mode instead of averaging two
  "huber": lambda e: F.huber_loss(e, torch.zeros_like(e), reduction="none", delta=0.1),
}

# the space the residual is measured in. Linear charges a 0.5 m error the same at 4 m as at 1 m,
# where only the near one decides whether the robot can step; log and inverse both make it relative,
# and inverse is what a monocular estimate is defined up to, which is how add_depth_noise degrades
DEPTH_SPACES: dict[DepthSpace, Callable] = {
  "linear": lambda d: d,
  "log": lambda d: torch.log(d.clamp(min=0.02)),  # 8 cm of a 4 m box, so the range stays bounded
  "inverse": lambda d: 1.0 / d.clamp(min=0.05),
}


def vvae_loss(out, x, beta=1.0, observed=None, mask=None, kl_floor=0.1, occlusion=0.0,
              residual="mse", depth_space="linear"):
    """Recon + KL, both summed per sample then meaned over the batch.

    mask drops target pixels, [B, H, W]. observed is [B, *latent_grid] in [0, 1] and scales the KL
    per latent voxel: kl_floor where seen, full beta where not.

    occlusion charges for density where the render's transmittance has already collapsed. An empty
    volume wins it outright, so read lift_agreement beside it.

    residual and depth_space set what an uncertain ray is charged for. See RESIDUALS above."""
    warp = DEPTH_SPACES[depth_space]
    err = RESIDUALS[residual](warp(out.x_hat) - warp(x))
    recon = (err if mask is None else err * mask.unsqueeze(1)).flatten(1).sum(1)
    kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
    if observed is not None:
        weight = observed.mul(kl_floor - 1).add(1.0)  # kl_floor where seen, 1 where not
        kl = kl * weight.unsqueeze(1)
    total = recon + beta * kl.flatten(1).sum(1)
    if occlusion and out.occluded is not None: total = total + occlusion * out.occluded
    return total.mean(), recon.mean(), kl.flatten(1).sum(1).mean()


SCHEDULERS = get_args(Scheduler)


PER_EPOCH = (PeriodicRestarts,)  # stepped once a validation, not once a batch

# how much of each epoch's score the selection average takes; at 0.2 it keeps about a third of a
# single validation's spread, so -best.pt tracks where the model is rather than which epoch was lucky
SELECT_EMA = 0.2


def make_scheduler(name, optimizer, lr, total_steps, period=50, warmup=0, shape="cosine"):
    """onecycle warms up over the first 30% then anneals to nothing, so it has to be told how long
    the run is. periodic anneals over a fixed period instead and begins again from a lower ceiling,
    so it never needs one and a run can be extended by asking for more epochs."""
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    if name == "periodic":
        # warmup is per cycle here, not per run, so it is a handful of epochs rather than a third
        return PeriodicRestarts(optimizer, lr, period=period, warmup=warmup or 5, shape=shape)
    return None


def checkpoint(model, optimizer, sched, config, epoch, global_step, score, smoothed):
    """Everything needed to carry on: weights, optimiser and schedule state, and where we were.
    Both scores travel: the epoch's own for reading, the smoothed one for choosing."""
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": sched.state_dict() if sched else None, "epoch": epoch,
            "global_step": global_step, "config": config, "score": score, "smoothed": smoothed}


def write_checkpoints(state, name, best_score, snapshot_every=0):
    """Latest, best and a periodic snapshot; returns the best score so far. Lower is better.
    Latest alone loses the good state whenever a run regresses, which they do.

    Best is chosen on the smoothed score, never the epoch's own. A single validation swings about
    as far as a whole run improves, so picking on it picks the luckiest draw rather than the best
    model. See .agents/learned.md."""
    state["tag"] = name  # so a --resume can find its way back to this lineage
    torch.save(state, CHECKPOINTS / f'{name}.pt')
    if state["smoothed"] < best_score:
        best_score = state["smoothed"]
        torch.save(state, CHECKPOINTS / f'{name}-best.pt')
        print(f'  new best, {best_score:.1f} smoothed ({state["score"]:.1f} this epoch)', flush=True)
    if snapshot_every and state["epoch"] % snapshot_every == 0:
        torch.save(state, CHECKPOINTS / f'{name}-e{state["epoch"]}.pt')
    return best_score


def restore(path, name, model, optimizer, sched, revive=True):
    """Put a run back where it was; returns (global_step, epoch, best_score, smoothed). Weights
    alone are not enough, since a schedule mid-cycle needs its own state back, and so does the
    average the best checkpoint is chosen on -- restarting that would let the first epoch after a
    resume win on its own.

    No horizon to repair here: a cycle is a fixed span, so continuing costs nothing but more of
    them."""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if sched and ckpt.get("scheduler"): sched.load_state_dict(ckpt["scheduler"])
    if isinstance(sched, PeriodicRestarts) and revive:
        sched.revive()  # a resume means there is more to try, not less
    best = CHECKPOINTS / f'{name}-best.pt'  # ckpt holds the last score, which need not be the best
    prior = torch.load(best, weights_only=False) if best.exists() else {}
    # checkpoints from before selection was smoothed carry only the raw score; fall back to it so
    # a run started under the old scheme still resumes, rather than dying on a missing key
    best_score = prior.get("smoothed") or prior.get("score", float("inf"))
    print(f'resumed {path} at epoch {ckpt["epoch"]}, lr {optimizer.param_groups[0]["lr"]:.2e}, '
          f'best {best_score:.1f}', flush=True)
    return ckpt["global_step"], ckpt["epoch"], best_score, ckpt.get("smoothed")


@torch.no_grad()
def observed_weight(model, context, context_poses, kl_mode, carved=None):
    """[B, *latent_grid] of how much of each latent voxel's region was seen, or None for a flat KL.
    Pooled down from the carving by the same factor the encoder downsamples by, so one weight lands
    on one latent voxel."""
    if kl_mode != "observed":
        return None
    seen = (model.encoder.carve(context, context_poses) if carved is None else carved)[:, 1:]
    shrink = model.config.downsample_scale ** model.config.n_downsamples
    return F.avg_pool3d(seen, shrink)[:, 0]


def train_geometric(model, train_loader, test_loader, name=None, epochs=20, lr=0.001, beta=1.0,
                    device='cuda', n_epochs=5, pose_noise=0.02, depth_noise=0.0, warmup=0,
                    project="vvae",
                    scheduler="none", shape="cosine", period=50, target="rgb", scenes=3, kl_mode="uniform", mask_unknown=False, unknown_weight=1.0, occlusion=0.0, residual="mse", depth_space="linear",
                    resume=None, taper=False, revive=True, snapshot_every=50):
    dataset = train_loader.dataset.dataset  # Subset -> the dataset it indexes into
    assert dataset.camera == model.camera, f"dataset {dataset.camera} != model {model.camera}"

    config = {  # names match sweep.json's parameters, so an agent's values land on the same keys
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "lr": lr,
        "beta": beta,
        "kl_mode": kl_mode,
        "mask_unknown": mask_unknown,
        "predict_unknown": model.config.predict_unknown,
        "scheduler": scheduler,
        "shape": shape,
        "period": period,
        "target": target,
        "pose_noise": pose_noise,
        "depth_noise": depth_noise,
        "occlusion": occlusion,
        "residual": residual,
        "depth_space": depth_space,
        "learned_lift": model.config.learned_lift,
        "steps_per_epoch": len(train_loader),
        "camera_fovy": dataset.camera[1],
        "num_frames": dataset.num_frames,
        "environments": sorted({environment_of(p) for p in train_loader.dataset.trajectory_paths}),
        "val_env": sorted({environment_of(p) for p in test_loader.dataset.trajectory_paths}),
        "head_jitter": dataset.include_head_jitter,
        "grid": model.config.grid_resolution,
        "latent": model.config.latent_shape,
        "base_channels": model.config.base_channels,
        "groups": model.config.groups,
        "enc_params": sum(p.numel() for p in model.encoder.parameters()),
        "camera": model.camera,
        "model": model.__class__.__name__,
        "optimizer": "Adam",
    }

    CHECKPOINTS.mkdir(exist_ok=True)
    with wandb.init(project=project, name=name, config=config, notes=METRIC_NOTES) as run:
        # wandb names a run for us and never repeats one, so checkpoints cannot collide. An offline
        # run has no name though, and every one of them would land on None.pt -- hence the id
        tag = name or run.name or run.id
        if resume and not name:  # a continuation belongs on the run it continues, or its snapshots
            prior = torch.load(resume, weights_only=False).get("tag")  # scatter across fun names
            if prior: tag = prior
        # the name says nothing about the run, so what it was lives beside it: one small json each,
        # greppable, so `grep -l '"occlusion": 1000' .runs/*.json` finds it without loading 14 MB
        (CHECKPOINTS / f'{tag}.json').write_text(json.dumps(config, indent=1, default=str))
        print(f'{tag}: {json.dumps(config, default=str)}', flush=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sched = make_scheduler(scheduler, optimizer, lr, epochs * len(train_loader), period,
                               warmup, shape)
        per_batch_lr = sched is not None and not isinstance(sched, PER_EPOCH)

        # epoch as its own axis; runs with different steps_per_epoch only align on it
        run.define_metric("epoch")
        for section in ("val/*", "train/recon_epoch", "train/lr"):
            run.define_metric(section, step_metric="epoch")
        # wandb keeps the running best itself, under a suffixed key: val/lift_agreement.max and
        # val/scene_corr.min, which is what a sweep should be pointed at
        run.define_metric("val/lift_agreement", step_metric="epoch", summary="max")
        run.define_metric("val/scene_corr", step_metric="epoch", summary="min")
        run.define_metric("val/recon_answerable", step_metric="epoch", summary="min")
        run.define_metric("val/selection_score", step_metric="epoch", summary="min")
        run.define_metric("val/depth_mae_m", step_metric="epoch", summary="min")
        run.define_metric("val/skill", step_metric="epoch", summary="max")
        run.define_metric("val/skill_answerable", step_metric="epoch", summary="max")
        run.define_metric("val/skill_vs_copy", step_metric="epoch", summary="max")
        run.define_metric("val/unknown_gap", step_metric="epoch", summary="max")
        run.define_metric("val/lift_fidelity", step_metric="epoch")
        run.define_metric("val/unknown_mae", step_metric="epoch", summary="min")
        run.define_metric("val/invented_density", step_metric="epoch", summary="min")
        run.define_metric("val/occluded_density", step_metric="epoch", summary="min")

        # two fixed val scenes, picked by index so they are identical in every run and comparable
        # across them, plus one drawn fresh from training each time it is logged. The fixed pair
        # shows whether the model is improving; the roving one shows what it is being shown, which
        # a single held-out building cannot
        val_set, train_set = test_loader.dataset, train_loader.dataset
        picks = torch.linspace(0, len(val_set) - 1, max(scenes - 1, 1)).round().long().tolist()
        samples = [tuple(x.unsqueeze(0) for x in val_set[i]) for i in picks]

        def roving():
            i = int(torch.randint(len(train_set), (1,)))
            return tuple(x.unsqueeze(0) for x in train_set[i]), i

        print(f"logging val scenes {picks} of {len(val_set)}, plus one drawn from training",
              flush=True)
        global_step, first_epoch, best_score, smoothed = 1, 0, float("inf"), None
        if resume:
            global_step, first_epoch, best_score, smoothed = restore(resume, tag, model,
                                                                     optimizer, sched, revive)
        if taper and isinstance(sched, PeriodicRestarts):
            torch.save(checkpoint(model, optimizer, sched, config, first_epoch, global_step,
                                  best_score, smoothed), CHECKPOINTS / f'{tag}-pretaper.pt')
            sched.taper(over=epochs)  # one-way, hence the snapshot above
            print(f'tapering to {sched.floor:.0e} over {epochs} epochs', flush=True)

        floors = baselines(model, test_loader, device, target)  # the zero points for val/skill*
        whole, mean_seen, copy_seen, graded = floors
        print(f'baselines over {graded:.0%} of the frame: mean image {mean_seen:.1f} '
              f'(whole frame {whole:.1f}), copy the anchor {copy_seen:.1f}', flush=True)

        for epoch in range(first_epoch, first_epoch + epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{first_epoch + epochs}')
            train_recon, train_batches = 0.0, 0

            for i, (frames, poses) in enumerate(pbar, 1):
                poses = add_pose_noise(poses, pose_noise)  # the pose quality deployment will have
                context, context_poses, truth, query = split_context_target(frames, poses, mode=target)
                optimizer.zero_grad()

                context, context_poses = context.to(device), context_poses.to(device)
                truth, query = truth.to(device), query.to(device)
                seen = add_depth_noise(context, depth_noise) if depth_noise else context
                # carved first, so its observedness can gate the occlusion charge. Built from the
                # clean context like the unknown head's target: it is supervision, not an input
                needs_carve = mask_unknown or kl_mode == "observed" or model.config.predict_unknown
                carved = model.encoder.carve(context, context_poses) if needs_carve else None
                out = model(seen, context_poses, query,
                            gate=carved[:, 1:2] if carved is not None else None)
                mask = (answerable_mask(carved, truth, query, model.config.bounds, *model.camera)
                        if mask_unknown and carved is not None else None)
                t_loss, t_recon, t_kl = vvae_loss(
                    out, truth, beta,
                    observed_weight(model, context, context_poses, kl_mode, carved), mask,
                    occlusion=occlusion, residual=residual, depth_space=depth_space)
                if model.config.predict_unknown and carved is not None:
                    # the unknown head, supervised everywhere against the carving
                    t_loss = t_loss + unknown_weight * F.mse_loss(
                        out.volume[:, -1], carved[:, 1], reduction="none").flatten(1).sum(1).mean()
                t_loss.backward()
                optimizer.step()
                if sched and not isinstance(sched, PER_EPOCH): sched.step()

                train_recon, train_batches = train_recon + t_recon.item(), train_batches + 1
                pbar.set_postfix(loss=f'{t_loss.item():.3g}', recon=f'{t_recon.item():.3g}', kl=f'{t_kl.item():.3g}')
                step_log = {"train/loss": t_loss.item(), "train/recon": t_recon.item(),
                            "train/kl": t_kl.item(), "epoch": epoch + i / len(train_loader)}
                if out.occluded is not None: step_log["train/occluded"] = out.occluded.mean().item()
                if per_batch_lr:  # a metric-driven schedule only moves once an epoch
                    step_log["train/lr"] = optimizer.param_groups[0]["lr"]
                run.log(step_log, step=global_step)
                global_step += 1

            metrics = validate(model, test_loader, device, target, beta, floors, vvae_loss)
            metrics |= {"train/recon_epoch": train_recon / max(train_batches, 1),
                        "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1}

            # a masked run optimises the answerable pixels, so that is what selection must read;
            # scoring the whole frame reads error the gradients never touched
            score = metrics["val/recon_answerable" if mask_unknown else "val/recon"]
            smoothed = score if smoothed is None else (1 - SELECT_EMA) * smoothed + SELECT_EMA * score
            metrics["val/selection_score"] = smoothed  # logged, so the chart shows what chose
            print(summarise(epoch + 1, metrics), flush=True)
            run.log(metrics, step=global_step)

            if isinstance(sched, PER_EPOCH): sched.step(score)
            state = checkpoint(model, optimizer, sched, config, epoch + 1, global_step, score,
                               smoothed)
            best_score = write_checkpoints(state, tag, best_score, snapshot_every)

            if epoch % n_epochs == 0:
                sample, index = roving()
                shown = [(f"scene{s}", x, "") for s, x in enumerate(samples)]
                shown.append(("roving", sample, f"training window {index}"))
                for name, sample, note in shown:
                    log_context_strip(model, sample, step=global_step, mode=target,
                                      key=f"media/context/{name}", note=note)
                    log_novel_view(model, sample, step=global_step, mode=target,
                                   key=f"media/novel/{name}", note=note)
                    log_pose_sweep(model, sample, step=global_step, mode=target,
                                   key=f"media/sweep/{name}", note=note)
                    log_latent(model, sample, step=global_step, mode=target,
                               key=f"media/latent/{name}", note=note)
                save_latent(model, samples[0], CHECKPOINTS / f'{tag}-latent.npz', mode=target)
    return model
