## License

This project is licensed under the MIT License.
See the [LICENSE](./LICENSE) file for more details.

# Using π0 model to drive Franka manipulator in Genesis Sim
* 1st goal: can it perform a trained task
* 2nd goal: can it learn a new task

## Assets
- DROID dataset: https://droid-dataset.github.io/droid/the-droid-dataset
    - More info: https://huggingface.co/KarlP/droid

- https://huggingface.co/datasets/haosulab/ReplicaCAD
    - It's a cleaned up/corrected version of https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset
    - Attributes schema: https://aihabitat.org/docs/habitat-sim/attributesJSON.html

```text
frl_apartment_basket_cv_decomp.glb has multiple convex-hull sub-meshes that Genesis doesn't natively read as individual pieces.
Genesis doesn’t automatically use each sub-mesh as its own collider. Instead, it:
	1.	merges them,
	2.	simplifies geometry,
	3.	and creates convex collision shapes.

[ ] load non-decomp GLB and let Genesis perform decomposition?
```

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
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=s3://openpi-assets/checkpoints/pi0_fast_droid
```

## Systems
Copying to openpi dir, which is mounted inside of
```bash
# Restart GNOME and DCV server
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-54-89-87-43.compute-1.amazonaws.com 'sudo systemctl restart gdm3 && sudo systemctl restart dcvserver'
# Start DCV session on Macbook
ec2-54-89-87-43.compute-1.amazonaws.com:8443#console
# Rsync code
rsync -avz --progress \
    --exclude '.git*' --exclude 'venv' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD/" \
    ubuntu@ec2-54-89-87-43.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-main/openpi/
```

```bash
xhost +local:root
# First time you run it may take a few minutes to load all assets into the sim. Subsequent should be much faster, under one min or so.
python pick_up_bottle.py
```

## Live CLI Debugger
```python
import IPython
IPython.embed()
```
