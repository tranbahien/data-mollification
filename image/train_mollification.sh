export CUDA_VISIBLE_DEVICES=2
python experiment.py \
--out_dir ./exp/mollification \
--mode mollification \
--mollification_iter 30000 \
--vanilla_iter 20000 \
--noise_start 0 \
--noise_end 3 \
--noise_tau 0.7