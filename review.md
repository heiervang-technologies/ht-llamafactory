Great PR! The graceful shutdown logic via `TrainerCallback` is a very clever way to cleanly interrupt training and preserve state without waiting for the next `save_steps`. Bumping the `save_total_limit` to prevent the interrupted checkpoint from evicting a normal checkpoint is a nice touch.

I noticed one issue that could cause failures in a multi-process/Distributed Data Parallel (DDP) setup:

**Race condition in `on_save` for multi-GPU training:**
In Hugging Face `Trainer`, the `on_save` callback executes on **all ranks** after the checkpoint is saved. If multiple processes hit `os.rename(src, dst)` concurrently, the first rank will successfully rename the directory, and the other ranks will raise a `FileNotFoundError` (or `[Errno 2] No such file or directory`) because `src` will no longer exist.

To fix this, you can use the standard Hugging Face `args.should_save` property, which is only `True` on the main process (rank 0), to ensure only one process performs the rename:

```python
    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self._interrupted:
            # Restore original save_total_limit
            if self._original_save_total_limit is not None:
                args.save_total_limit = self._original_save_total_limit
                self._original_save_total_limit = None
                
            # Rename checkpoint so it is excluded from future rotation
            src = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            dst = os.path.join(args.output_dir, f"interrupt-{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            # Ensure renaming only happens on the main process to avoid FileNotFoundError in DDP
            if args.should_save and os.path.isdir(src):
                os.rename(src, dst)
                logger.info_rank0(f"Interrupt checkpoint saved at: {dst}")
```

Other than that, the integration with `FinetuningArguments` and the custom PPO training loop looks solid, and the tests clearly validate the behavior. Once the race condition is addressed, this will be good to go!
