# Using π0 model to drive Franka manipulator in Genesis Sim
* 1st goal: can it perform a trained task
* 2nd goal: can it learn a new task

## TODO
[ ] Run Genesis and OpenPi in separate containers -- communicate via default network ports or create new SHM for faster communication.
    < > If SHM, profile transfer time before and after

## Reproducibility
Current versions being used:
- [Genesis-5cc3d5](https://github.com/Genesis-Embodied-AI/Genesis/commit/5cc3d5606c3c1e08eb3c628957e76e8e8512ae13)
- [OpenPi-92b108](https://github.com/Physical-Intelligence/openpi/commit/92b10824421d6d810eb1e398330acd79dc7cd934)

### Start OpenPi local server
Start model server
- pi0_fast_droid: Autoregressive π0-FAST-DROID model
- pi0_droid: Diffusion π0-DROID model

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=s3://openpi-assets/checkpoints/pi0_fast_droid
```

## Systems
Copying to openpi dir, which is mounted inside of
```bash
rsync -avz --progress \
    --exclude '.git*' --exclude 'venv' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD/" \
    ubuntu@ec2-34-207-228-162.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-main/openpi/
```

## Live CLI Debugger
```python
import IPython
IPython.embed()
```