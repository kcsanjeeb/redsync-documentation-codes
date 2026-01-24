python -u -m src.train_freeze_phase2_async_fixed_fc_reuse_gated_v4 \
  --epochs 200 \
  --bootstrap_epochs 120 \
  --probe_every_epochs 1 \
  --log_path runs/phase_2/phase_2C/v4_cache_gated.jsonl \
  --feature_cache \
  --fc_codec zstd \
  --fc_level 3 \
  --fc_max_items 256