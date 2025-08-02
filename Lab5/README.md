# Lab 5

## Training
Train:  
```
python enhanced_dqn.py
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
--model-path best_model.pt
--env (CartPole-v1 or ALE/Pong-v5)
--output-dir ./eval_results
--save-video
```
