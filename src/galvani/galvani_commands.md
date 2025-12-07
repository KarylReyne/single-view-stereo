### Galvani a100
srun --job-name "eval01" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:4 --time 1:00:00 --pty bash