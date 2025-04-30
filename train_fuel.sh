python3 train.py --model ppo_sb3 --num-airplanes 1 \
 --outdir new_logs/fuel \
 --max-episodes 500 \
 --max-steps-per-episode 50000 \
 --save-freq 20 \
 --eval-freq 2 \
 --eval-episodes 5 \
 --threads 10 \
 --starting-fuel 10000 \
 --reduced-time-penalty \
 --wind-badness 10 \
 --scenario FuelScenario

