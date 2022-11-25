#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M  
#SBATCH --account=def-bpirenne
#SBATCH --job-name=test

echo 'Loading libraries'
module load python/3.9
module load cuda/11.1 
module load cudnn
echo '...done'

echo 'Copy data tmp dir'
cp /scratch/xmouy/ketos_databases/fish_bc/spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700/database.h5 $SLURM_TMPDIR/
echo '...done'

CONFIG_DIR=/scratch/xmouy/ketos_databases/fish_bc/spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700
OUT_DIR=/scratch/xmouy/results/FS-400ms-bs16-ep50-norm_test

source /home/xmouy/ketos262-env/bin/activate

echo 'Start training'
python3 ./train_ketos.py --batch_size=128 --n_epochs=20 --db_file=$SLURM_TMPDIR/database.h5 --recipe_file=$CONFIG_DIR/recipe.json --spec_config_file=$CONFIG_DIR/spec_config.json --out_dir=$OUT_DIR --checkpoints_dir=$OUT_DIR --logs_dir=$OUT_DIR
echo '...done'