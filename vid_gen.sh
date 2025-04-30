# python3 replay.py \
#  --model ppo_sb3 \
#  --curr-stage-entry-point max \
#  --curr-stages 50 \
#  --skip-frames 100 \
#  --episodes 1 \
#  --sleep 1 \
#  --scenario MvaGoAroundScenario \
#  --num-airplanes 1 \
#  --mp4 \
#  --starting-fuel 200 \
#  --wind-badness 10 \
#   --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/fuel_model_200_fuel/sb3_ppo_model_160.zip \
#   --savetraj
# echo "vid1: done"
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
 --starting-fuel 1000 \
 --wind-badness 10 \
  --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/bad_fuel/sb3_ppo_model_120.zip \
  --savetraj
# echo "vid2: done"
# python3 replay.py \
#  --model ppo_sb3 \
#  --curr-stage-entry-point max \
#  --curr-stages 50 \
#  --skip-frames 100 \
#  --episodes 1 \
#  --sleep 1 \
#  --scenario MvaGoAroundScenario \
#  --num-airplanes 1 \
#  --mp4 \
#  --starting-fuel 10000 \
#  --wind-badness 10 \
#   --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/goaround_sb3_ppo_model_120.zip \
#   --savetraj

python3 replay.py \
 --model ppo_sb3 \
 --curr-stage-entry-point max \
 --curr-stages 100 \
 --skip-frames 50 \
 --episodes 1 \
 --sleep 1 \
 --scenario SupaSupa \
 --num-airplanes 1 \
 --mp4 \
 --starting-fuel 10000 \
 --wind-badness 5 \
  --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/cute_model_supasupa_trainedtill_stage98/curr_model_stage98_entry114.35479633652099_81.02785148201173_hdg247_ep40.zip \
  --savetraj

# python3 replay.py \
#  --model ppo_sb3 \
#  --curr-stage-entry-point max \
#  --curr-stages 100 \
#  --skip-frames 100 \
#  --entry 80,100 \
#  --level 250 \
#  --heading 135 \
#  --episodes 1 \
#  --sleep 1 \
#  --scenario SupaSupa \
#  --num-airplanes 1 \
#  --mp4 \
#  --starting-fuel 10000 \
#  --wind-badness 5 \
#   --checkpoint /Users/pjr/Desktop/Plaksha/sem6/RL/project/atc-lite/final_models/cute_model_supasupa_trainedtill_stage98/curr_model_stage98_entry114.35479633652099_81.02785148201173_hdg247_ep40.zip \
#   --savetraj