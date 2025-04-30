#!/bin/bash

# Script to train a generic model without curriculum learning
# Uses standard PPO, cycling through 5 entry points from the last curriculum stage every 10 episodes

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="new_logs/generic_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Run the training with simplified configuration
python train.py \
  --outdir "${LOG_DIR}" \
  --model ppo_sb3 \
  --scenario ModifiedLOWW \
  --generic-training \
  --reduced-time-penalty \
  --max-episodes 10000 \
  --max-steps-per-episode 50000 \