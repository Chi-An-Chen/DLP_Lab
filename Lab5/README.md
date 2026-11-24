# Lab 5

Work from `Lab5/`. Select the same environment and model variants for training and evaluation; evaluation requires a checkpoint from that run.

## Training
Train:  
```
python DQN.py
```
```
--env (CartPole-v1 or ALE/Pong-v5)
--use-dueling
--use-double-dqn
--use-prioritized-replay
--save-dir ./enhanced_results
```
(Make sure to edit the path for the dataset or checkpoint path etc.)

## Testing
```
python test_model.py
```
```
--model-path /path/to/checkpoint.pt
--env (CartPole-v1 or ALE/Pong-v5)
--output-dir ./eval_results
--save-video
```
