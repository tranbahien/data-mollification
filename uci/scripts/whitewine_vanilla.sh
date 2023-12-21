export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 1 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/vanilla/seed_1 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 1 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/vanilla/seed_1 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 1 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/vanilla/seed_1 --epochs 150 --lr 1e-4 --mode vanilla 

export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 2 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/vanilla/seed_2 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 2 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/vanilla/seed_2 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 2 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/vanilla/seed_2 --epochs 150 --lr 1e-4 --mode vanilla 

export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 3 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/vanilla/seed_3 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 3 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/vanilla/seed_3 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 3 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/vanilla/seed_3 --epochs 150 --lr 1e-4 --mode vanilla 

export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 4 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/vanilla/seed_4 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 4 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/vanilla/seed_4 --epochs 150 --lr 1e-4 --mode vanilla 
export CUDA_VISIBLE_DEVICES=3 && python main.py --seed 4 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/vanilla/seed_4 --epochs 150 --lr 1e-4 --mode vanilla 