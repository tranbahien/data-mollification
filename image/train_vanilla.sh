export CUDA_VISIBLE_DEVICES=3
python experiment.py \
--out_dir ./exp/vanilla \
--mode vanilla \
--mollification_iter 30000 \
--vanilla_iter 20000