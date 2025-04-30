python3 replay.py --model ppo_sb3 \
 --starting-fuel 2000 \
 --wind-badness 10 \
 --scenario MvaGoAroundScenario \
 --wind-dirn 300 \
 --checkpoint "new_logs/fuel/sb3_ppo_model_120.zip" \
 --skip-frames 100\