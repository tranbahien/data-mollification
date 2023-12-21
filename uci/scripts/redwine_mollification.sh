export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 1 --flow maf --dataset REDWINE --out_dir exp/maf/REDWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 1 --flow realnvp --dataset REDWINE --out_dir exp/realnvp/REDWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 1 --flow glow --dataset REDWINE --out_dir exp/glow/REDWINE/mollification/seed_1 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 2 --flow maf --dataset REDWINE --out_dir exp/maf/REDWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 2 --flow realnvp --dataset REDWINE --out_dir exp/realnvp/REDWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 2 --flow glow --dataset REDWINE --out_dir exp/glow/REDWINE/mollification/seed_2 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 3 --flow maf --dataset REDWINE --out_dir exp/maf/REDWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 3 --flow realnvp --dataset REDWINE --out_dir exp/realnvp/REDWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 3 --flow glow --dataset REDWINE --out_dir exp/glow/REDWINE/mollification/seed_3 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 4 --flow maf --dataset REDWINE --out_dir exp/maf/REDWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 4 --flow realnvp --dataset REDWINE --out_dir exp/realnvp/REDWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150
export CUDA_VISIBLE_DEVICES=5 && python main.py --seed 4 --flow glow --dataset REDWINE --out_dir exp/glow/REDWINE/mollification/seed_4 --epochs 150 --lr 1e-4 --mode mollification  --mollification_epochs 150

