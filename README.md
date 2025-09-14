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
```

## TODO
[ ] Run Genesis and OpenPi in separate containers -- communicate via default network ports or create new SHM for faster communication.
    < > If SHM, profile transfer time before and after

## Reproducibility
Current versions being used:
- [Genesis-5cc3d5](https://github.com/Genesis-Embodied-AI/Genesis/commit/5cc3d5606c3c1e08eb3c628957e76e8e8512ae13)
    - `docker build -t genesis:5cc3d5 -f docker/Dockerfile docker`
- [OpenPi-92b108](https://github.com/Physical-Intelligence/openpi/commit/92b10824421d6d810eb1e398330acd79dc7cd934)
    - Latest [OpenPi-df866f](https://github.com/Physical-Intelligence/openpi/tree/df866f61f95d801504adda66f412e1ef4bf7734c)
```bash
# First Time
docker compose -f scripts/docker/compose.yml up --build
    # Edit scripts/docker/compose.yml image name to openpi:b84cc7 to create the openpi_jointpos_b84cc7 container
# Once server starts hit ctrl+c and run the following
# Subsequent runs
docker compose -f scripts/docker/compose.yml run -d --name openpi_jointpos_b84cc7 \
    openpi_server bash -lc "tail -f /dev/null"
docker exec -it openpi_jointpos_b84cc7 /bin/bash
```
```bash
# openpi-karl-droid_policies: https://github.com/Physical-Intelligence/openpi/tree/karl/droid_policies (literal commit used: https://github.com/Physical-Intelligence/openpi/tree/b84cc75031eb3a9cbcfb1d55ee85fbd7db81e8bb)

# Running model
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```

- Latest [Genesis-66708b](https://github.com/Genesis-Embodied-AI/Genesis/tree/66708b2df7b2909b59915852e015ea1bb91bb948)
    - `docker build -t genesis:66708b -f docker/Dockerfile docker`
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
            --name genesis-66708b \
            --network host \
            genesis:66708b

        # Restart
        docker start genesis-e064db
        docker exec -it genesis-e064db /bin/bash

        # on mac
        rsync -avz --progress \
            --exclude '.git*' --exclude 'venv' --exclude '__pycache__' \
            -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
            "$PWD" \
            ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com:/home/ubuntu/dev/
        ```
        ```bash
        # In Genesis-e064db dir:
        mkdir ext
        cd ext
        git clone git@github.com:Physical-Intelligence/openpi.git  # git-hash df866f or karl's version
        cd openpi  # cd openpi-karl-droid_policies
        pip install -e . --no-deps && pip install -e packages/openpi-client  # If you get an issue about dependencies in this case those can be ignored
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
```bash
# Now that there's a version of `pi0_fast_droid` trained with joint position actions, it's much
# easier to simulate. Before with orig `pi0_fast_droid` we can get the gripper to for example
# get near an object but it cannot pick it up.
# https://github.com/Physical-Intelligence/openpi/blame/main/examples/droid/README_train.md#L43-L45
# https://github.com/arhanjain/sim-evals/commit/171711551581955dcfa017ad0156e2497c659537


```

## Systems
Copying to openpi dir, which is mounted inside of
```bash
# Restart GNOME and DCV server
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com 'sudo systemctl restart gdm3 && sudo systemctl restart dcvserver'
# Start DCV session on Macbook
ec2-44-200-228-145.compute-1.amazonaws.com:8443#console

# Rsync code # Move ReplicCAD assets
rsync -avz --progress \
    --exclude '.git*' --exclude 'venv' --exclude '__pycache__' --delete \
    -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD" \
    ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com:/home/ubuntu/dev/

rsync -avz --progress \
    --exclude '.git*' --exclude 'venv' --exclude '__pycache__' --delete \
    -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD/" \
    ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-e064dbc/dev/
```

**More automated:**
```bash
# Run through ssh
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com
sudo systemctl restart gdm3 && sudo systemctl restart dcvserver
# Enter desktop using DCV: ec2-44-200-228-145.compute-1.amazonaws.com:8443#console, then move on to next steps
# ./enter_genesis.sh
./Desktop/Genesis-main/openpi/enter_genesis.sh
python openpi/pick_up_bottle.py
```

```bash
# Run through ssh
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com
# Not needed anymore: sudo systemctl restart gdm3 && sudo systemctl restart dcvserver
# Enter desktop using DCV: ec2-44-200-228-145.compute-1.amazonaws.com:8443#console, then move on to next steps
who  # get user DISPLAY, e.g. ":1"
export DISPLAY=:1
xhost +local:root
. /home/ubuntu/dev/explorations/sys_scripts/gnome_view_hw_metrics.sh
docker start genesis-66708b
docker start openpi_jointpos_b84cc7
docker exec -it genesis-66708b /bin/bash

ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com
docker exec -it openpi_jointpos_b84cc7 /bin/bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

python pick_up_bottle.py

```

```bash
# Proxy through ssh
ssh -i ~/.ssh/aws-us-east-1.pem -L 8443:localhost:8443 ubuntu@ec2-44-200-228-145.compute-1.amazonaws.com
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
- [Evaluation and discussion of π0-FAST-DROID model](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)

### Other Resources not used here, but helpful
- [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py), python package to import open-source robot descriptions
- [PickNikRobotics/ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/main), has assets for the Robotiq gripper, packaged for ROS2
