# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for graceful shutdown with checkpoint save on SIGINT/SIGTERM."""

import os
import signal
import threading

import pytest

from transformers import TrainerCallback

from llamafactory.train.tuner import run_exp


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "alpaca_en_demo",
    "dataset_dir": "REMOTE:" + DEMO_DATA,
    "template": "llama3",
    "cutoff_len": 1,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "report_to": "none",
}


class SendSIGINTCallback(TrainerCallback):
    """A callback that sends SIGINT at a specific training step.

    This triggers the GracefulStopCallback's signal handler deterministically,
    avoiding timing issues with threading.Timer.
    """

    def __init__(self, fire_after_step: int = 1) -> None:
        self._fire_after_step = fire_after_step
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self._fire_after_step and not self._fired:
            self._fired = True
            os.kill(os.getpid(), signal.SIGINT)


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_graceful_shutdown_saves_interrupt_checkpoint(tmp_path):
    """When save_on_interrupt=True and SIGINT is received, an interrupt checkpoint is saved."""
    output_dir = str(tmp_path / "interrupt_test")

    # The DelayedSIGINTCallback sends SIGINT after step 1.
    # The GracefulStopCallback (parent) catches it, finishes the step, saves, and exits.
    sigint_callback = SendSIGINTCallback(fire_after_step=1)
    run_exp(
        {
            **TRAIN_ARGS,
            "output_dir": output_dir,
            "max_steps": 200,
            "save_on_interrupt": True,
            "save_steps": 9999,  # no periodic saves
        },
        callbacks=[sigint_callback],
    )

    # An interrupt-checkpoint-* directory should exist
    entries = os.listdir(output_dir)
    interrupt_checkpoints = [e for e in entries if e.startswith("interrupt-checkpoint-")]
    assert len(interrupt_checkpoints) == 1, f"Expected 1 interrupt checkpoint, found: {entries}"

    # Verify the checkpoint contains model and optimizer state
    ckpt_dir = os.path.join(output_dir, interrupt_checkpoints[0])
    assert os.path.isdir(ckpt_dir)
    assert os.path.exists(os.path.join(ckpt_dir, "trainer_state.json"))
    # optimizer state is saved in optimizer.pt or optimizer/ dir
    has_optimizer = os.path.exists(os.path.join(ckpt_dir, "optimizer.pt")) or os.path.isdir(
        os.path.join(ckpt_dir, "optimizer")
    )
    assert has_optimizer, f"No optimizer state in {os.listdir(ckpt_dir)}"


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_no_interrupt_checkpoint_without_flag(tmp_path):
    """When save_on_interrupt=False (default), SIGINT kills training without saving."""
    output_dir = str(tmp_path / "no_interrupt_test")

    # Without save_on_interrupt, SIGINT should raise KeyboardInterrupt.
    # We send SIGINT via a threading timer since there's no GracefulStopCallback to hook into.
    def _send_sigint():
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(3.0, _send_sigint)
    timer.daemon = True
    timer.start()

    with pytest.raises(KeyboardInterrupt):
        run_exp(
            {
                **TRAIN_ARGS,
                "output_dir": output_dir,
                "max_steps": 200,
                "save_steps": 9999,
            }
        )

    # No interrupt checkpoint should exist
    if os.path.isdir(output_dir):
        entries = os.listdir(output_dir)
        interrupt_checkpoints = [e for e in entries if e.startswith("interrupt-checkpoint-")]
        assert len(interrupt_checkpoints) == 0, f"Should have no interrupt checkpoints, found: {entries}"


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_interrupt_checkpoint_does_not_count_towards_save_total_limit(tmp_path):
    """The interrupt checkpoint should not evict regular checkpoints."""
    output_dir = str(tmp_path / "limit_test")

    # save_total_limit=1, save every 2 steps, interrupt after step 5
    sigint_callback = SendSIGINTCallback(fire_after_step=5)
    run_exp(
        {
            **TRAIN_ARGS,
            "output_dir": output_dir,
            "max_steps": 200,
            "save_on_interrupt": True,
            "save_steps": 2,
            "save_total_limit": 1,
        },
        callbacks=[sigint_callback],
    )

    entries = os.listdir(output_dir)
    interrupt_checkpoints = [e for e in entries if e.startswith("interrupt-checkpoint-")]
    regular_checkpoints = [e for e in entries if e.startswith("checkpoint-")]

    # Should have exactly 1 regular checkpoint (save_total_limit=1) + 1 interrupt checkpoint
    assert len(interrupt_checkpoints) == 1, f"Expected 1 interrupt checkpoint, found: {entries}"
    assert len(regular_checkpoints) == 1, f"Expected 1 regular checkpoint (save_total_limit=1), found: {entries}"
