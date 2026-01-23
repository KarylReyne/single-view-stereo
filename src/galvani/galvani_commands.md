### Galvani a100
srun --job-name "eval01" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:4 --time 1:00:00 --pty bash

## tmux
tmux ls <!-- list running jobs -->
tmux new -s SESSION_NAME <!-- create job -->
... <!-- run job (inside tmux terminal) -->
CTRL+B+D <!-- detach job (can also close the terminal via vsc) -->
tmux attach -t SESSION_NAME <!-- attach job -->
exit <!-- exit job (inside tmux terminal) -->
tmux kill-session -t SESSION_NAME <!-- terminate session -->
tmux kill-server <!-- kill/reset tmux server (in case it hangs) -->