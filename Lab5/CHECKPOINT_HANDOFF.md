# Reusing a Lab 5 checkpoint

Start in `Lab5/`. The [lab instructions](README.md) cover training and evaluation. Carry the environment ID and the checkpoint from the same run across the two scripts.

| What you have | Entry and required input | What you get |
| --- | --- | --- |
| A Gymnasium environment such as CartPole or ALE/Pong | [DQN.py](DQN.py) takes the environment ID via `--env` and a checkpoint directory via `--save-dir`. | Training can save `best_model.pt` or episode checkpoints named `model_ep<N>.pt` under that directory. |
| A saved checkpoint | [test_model.py](test_model.py) takes the checkpoint **file** via `--model-path` and the corresponding environment ID via `--env-name`. | The evaluator prints episode results; `--save-video` writes MP4s under `--output-dir`. |

For example, when evaluating a Pong checkpoint, pass `--env-name ALE/Pong-v5`; the evaluator otherwise defaults to CartPole.
