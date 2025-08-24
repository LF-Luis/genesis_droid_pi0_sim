## License

This project is licensed under the MIT License.
See the [LICENSE](./LICENSE) file for more details.

# Using π0 model to drive Franka manipulator in Genesis Sim
* 1st goal: can it perform a trained task
* 2nd goal: can it learn a new task

## Assets
- DROID dataset: https://droid-dataset.github.io/droid/the-droid-dataset
    - More info: https://huggingface.co/KarlP/droid
** Reproducing the DROID dataset setup:**
- Franka Emika Panda
- 2F-85 Robotiq Gripper
    - https://github.com/google-deepmind/mujoco_menagerie
        - https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85
        - https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4
- (scene cam) Zed 2 (match intrinsic values)
- (wrist cam) Zed Mini (match intrinsic values)

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
    - Latest [OpenPi-df866f](https://github.com/Physical-Intelligence/openpi/tree/df866f61f95d801504adda66f412e1ef4bf7734c)
        ```bash
        # First Time
        docker compose -f scripts/docker/compose.yml up --build
        # Once server starts hit ctrl+c and run the following
        # Subsequent runs
        docker compose -f scripts/docker/compose.yml run -d --name openpi \
            openpi_server bash -lc "tail -f /dev/null"
        docker exec -it openpi /bin/bash
        ```
- Latest [Genesis-e064dbc](https://github.com/Genesis-Embodied-AI/Genesis/tree/e064dbc8468d8fd47c0561218d8efd14565144c9)
    - `docker build -t genesis:e064dbc -f docker/Dockerfile docker`
        ```bash
        who
        export DISPLAY=:1
        xhost +local:root
        # First time build
        docker run --gpus all -dit \
            -e DISPLAY=$DISPLAY \
            -v /dev/dri:/dev/dri \
            -v /tmp/.X11-unix/:/tmp/.X11-unix \
            -v $PWD:/workspace \
            --name genesis \
            --network host \
            genesis:e064dbc

        # Restart
        docker start genesis-e064dbc
        docker exec -it genesis-e064dbc /bin/bash

        # on mac
        rsync -avz --progress \
            --exclude '.git*' --exclude 'venv' --exclude '__pycache__' \
            -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
            "$PWD/" \
            ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-e064dbc/luis_dev/
        ```
        ```bash
        # In Genesis-e064dbc dir:
        mkdir ext
        cd ext
        git clone git@github.com:Physical-Intelligence/openpi.git  # git-hash df866f
        cd openpi
        pip install -e . --no-deps
        pip install -e packages/openpi-client
        ```

### Start OpenPi local server
Start model server
- pi0_fast_droid: Autoregressive π0-FAST-DROID model
- pi0_droid: Diffusion π0-DROID model
```bash
docker exec -it openpi /bin/bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

## Systems
Copying to openpi dir, which is mounted inside of
```bash
# Restart GNOME and DCV server
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com 'sudo systemctl restart gdm3 && sudo systemctl restart dcvserver'
# Start DCV session on Macbook
ec2-3-90-146-100.compute-1.amazonaws.com:8443#console
# Rsync code
rsync -avz --progress \
    --exclude '.git*' --exclude 'venv' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD/" \
    ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-e064dbc/dev/
```

**More automated:**
```bash
# Run through ssh
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com
sudo systemctl restart gdm3 && sudo systemctl restart dcvserver
# Enter desktop using DCV: ec2-3-90-146-100.compute-1.amazonaws.com:8443#console, then move on to next steps
# ./enter_genesis.sh
./Desktop/Genesis-main/openpi/enter_genesis.sh
python openpi/pick_up_bottle.py
```

```bash
# Run through ssh
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com
sudo systemctl restart gdm3 && sudo systemctl restart dcvserver
# Enter desktop using DCV: ec2-3-90-146-100.compute-1.amazonaws.com:8443#console, then move on to next steps
who  # get user DISPLAY, e.g. ":1"
export DISPLAY=:1
xhost +local:root
docker start genesis
docker start openpi
docker exec -it genesis /bin/bash

ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com
docker exec -it openpi /bin/bash

python pick_up_bottle.py

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

```bash
# Proxy through ssh
ssh -i ~/.ssh/aws-us-east-1.pem -L 8443:localhost:8443 ubuntu@ec2-3-90-146-100.compute-1.amazonaws.com
# Start DCV session on Macbook
localhost:8443
```

```bash
xhost +local:root
docker start dev-genesis
docker exec -it dev-genesis /bin/bash
# First time you run it may take a few minutes to load all assets into the sim. Subsequent should be much faster, under one min or so.
python pick_up_bottle.py
```

## Live CLI Debugger
```python
import IPython
IPython.embed()
```

### Resources
- [Deepmind's mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), open-source robot sim assets

### Other Resources not used here, but helpful
- [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py), python package to import open-source robot descriptions
- [PickNikRobotics/ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/main), has assets for the Robotiq gripper, packaged for ROS2