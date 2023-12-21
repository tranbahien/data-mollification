export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 1 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 1 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 1 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 2 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 2 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 2 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 3 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 3 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 3 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 4 --flow maf --dataset WHITEWINE --out_dir exp/maf/WHITEWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 4 --flow realnvp --dataset WHITEWINE --out_dir exp/realnvp/WHITEWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=6 && python main.py --seed 4 --flow glow --dataset WHITEWINE --out_dir exp/glow/WHITEWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

