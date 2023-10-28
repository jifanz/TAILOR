# TAILOR

### Package Requirements
Install the following dependencies:

```PyTorch```

```Weights and Biases (wandb, register for user)```


### Run Experiments
For multi-label classification:
```
python mp_launcher_multi_label.py [YOUR_WANDB_NAME]
```
For multi-class classification:
```
python mp_launcher_multi_class.py [YOUR_WANDB_NAME]
```


### File Structure
* `main.py` Entry point to our process.
* `hyperparam.py` Hyper-parameter settings.
* `strategy/strategies.py` Entry point to meta algorithms.
* `strategy/meta/utils.py` Entry point to candidate algorithms.
* `strategy/meta/thompson_meta.py` TAILOR implementation.
* `dataset/datasets.py` Entry point to dataset loading.
