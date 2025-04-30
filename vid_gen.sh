python3 replay.py \
 --model ppo_sb3 \
 --curr-stage-entry-point max \
 --curr-stages 50 \
 --skip-frames 100 \
 --episodes 1 \
 --sleep 1 \
 --scenario MvaGoAroundScenario \
 --num-airplanes 1 \
 --mp4 \
 --starting-fuel 200 \
 --wind-badness 10 \
  --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/fuel_model_200_fuel/sb3_ppo_model_160.zip
echo "vid1: done"
python3 replay.py \
 --model ppo_sb3 \
 --curr-stage-entry-point max \
 --curr-stages 50 \
 --skip-frames 100 \
 --episodes 1 \
 --sleep 1 \
 --scenario MvaGoAroundScenario \
 --num-airplanes 1 \
 --mp4 \
 --starting-fuel 10000 \
 --wind-badness 10 \
  --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/fuel_model_bad_max_fuel/sb3_ppo_model_120.zip
echo "vid2: done"