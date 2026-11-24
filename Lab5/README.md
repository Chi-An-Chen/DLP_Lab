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

From `Lab5/`, evaluate a saved CartPole checkpoint and write videos to `eval_results/`:

```bash
python test_model.py --model-path ./enhanced_results/best_model.pt --env-name CartPole-v1 --output-dir ./eval_results --save-video
```

Use the checkpoint file from your own run. For a Pong checkpoint, set `--env-name ALE/Pong-v5`. See the [checkpoint handoff](CHECKPOINT_HANDOFF.md) for the training and evaluation paths.
