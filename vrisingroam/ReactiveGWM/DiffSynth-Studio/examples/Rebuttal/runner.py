"""Exact-step training runner for the rebuttal experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from diffsynth.diffusion.runner import (
    initialize_deepspeed_gradient_checkpointing,
)
from diffsynth.diffusion.training_module import DiffusionTrainingModule

from .checkpoint_io import RebuttalCheckpointLogger
from .trainable_policy import build_and_audit_optimizer


def first_item_collate(items):
    """The project uses per-process batch size one."""

    if len(items) != 1:
        raise ValueError(f"Expected batch size 1, got {len(items)}")
    return items[0]


@dataclass(frozen=True)
class TrainingResult:
    initial_step: int
    final_step: int
    dataloader_epochs: int


def _state_directory(output_path: str, step: int) -> Path:
    return Path(output_path) / f"state-{step}"


def launch_rebuttal_training(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: RebuttalCheckpointLogger,
    *,
    args,
    gradient_auditor: Optional[Callable[[torch.nn.Module, int], None]] = None,
) -> TrainingResult:
    """Train until exactly ``max_train_steps`` optimizer updates have completed."""

    max_train_steps = int(args.max_train_steps)
    if max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive")
    initial_step = int(model_logger.num_steps)
    if initial_step < 0 or initial_step >= max_train_steps:
        raise ValueError(
            f"Initial step {initial_step} must be in [0, {max_train_steps})"
        )

    optimizer = build_and_audit_optimizer(
        model,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    # Keep the requested 5e-5 learning rate from the first through final step.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    num_workers = int(args.dataset_num_workers)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=first_item_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if len(dataloader) == 0:
        raise ValueError("Training dataset is empty")

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    resume_state_path = getattr(args, "resume_state", None)
    if resume_state_path:
        if not os.path.isdir(resume_state_path):
            raise FileNotFoundError(
                f"--resume_state is not a directory: {resume_state_path}"
            )
        accelerator.load_state(resume_state_path)
        if accelerator.is_main_process:
            print(
                f"[resume-state] restored optimizer/scheduler/RNG/data state "
                f"from {resume_state_path}; next step={initial_step + 1}",
                flush=True,
            )

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    progress = tqdm(
        total=max_train_steps,
        initial=initial_step,
        disable=not accelerator.is_main_process,
        desc="optimizer steps",
    )
    try:
        while model_logger.num_steps < max_train_steps:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            saw_batch = False
            for data in dataloader:
                saw_batch = True
                with accelerator.accumulate(model):
                    if bool(getattr(dataset, "load_from_cache", False)):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    if not bool(torch.isfinite(loss.detach()).all()):
                        raise FloatingPointError(
                            "Non-finite loss before step "
                            f"{model_logger.num_steps + 1}: "
                            f"{loss.detach()}"
                        )
                    accelerator.backward(loss)

                    next_step = model_logger.num_steps + 1
                    if gradient_auditor is not None and accelerator.sync_gradients:
                        gradient_auditor(
                            accelerator.unwrap_model(model),
                            next_step,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue

                model_logger.on_step_end(
                    accelerator,
                    model,
                    args.save_steps,
                    loss=loss,
                )
                progress.update(1)

                if (
                    bool(getattr(args, "save_full_state", False))
                    and args.save_steps
                    and model_logger.num_steps % args.save_steps == 0
                ):
                    state_dir = _state_directory(
                        model_logger.output_path, model_logger.num_steps
                    )
                    if state_dir.exists() and not bool(
                        getattr(args, "allow_checkpoint_overwrite", False)
                    ):
                        raise FileExistsError(
                            f"Refusing to overwrite accelerator state: {state_dir}"
                        )
                    if accelerator.is_main_process:
                        state_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(str(state_dir))
                    if accelerator.is_main_process:
                        print(f"[save-state] wrote {state_dir}", flush=True)

                if model_logger.num_steps >= max_train_steps:
                    break
            if not saw_batch:
                raise RuntimeError("DataLoader yielded no batches")
            epoch += 1
    finally:
        progress.close()

    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.wait_for_everyone()
    return TrainingResult(
        initial_step=initial_step,
        final_step=model_logger.num_steps,
        dataloader_epochs=epoch,
    )


__all__ = [
    "TrainingResult",
    "first_item_collate",
    "launch_rebuttal_training",
]
