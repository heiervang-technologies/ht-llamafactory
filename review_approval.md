Thanks for implementing the fix so quickly! The use of `args.should_save` to coordinate the rename strictly on the main process perfectly resolves the race condition for multi-GPU training. 

The added warning for distributed training makes perfect sense too. LGTM!
