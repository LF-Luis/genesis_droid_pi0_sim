# Steps to setup GNOME desktop and DCV on AWS EC2 instance -- WIP

https://docs.aws.amazon.com/dcv/latest/userguide/client-mac.html
https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing.html



--------------------------------------------------------------------------------

| Instance Size   | GPU Memory | vCPUs | Memory  | Storage (GB)      | On Demand Price/hr |
|------------------|------------|-------|---------|--------------------|---------------------|
| g4dn.xlarge (us-west-2) | -          | 4     | 16 GiB  | 125 GB NVMe SSD    | $0.526              |
| g5.xlarge        | 1          | 4     | 16      | 1x250              | $1.006              |
| g6.xlarge        | 1          | 4     | 16      | 1x250              | $0.805              |
| g6e.xlarge       | -          | 4     | 32 GiB  | 1x250              | $1.861              |


--------------------------------------------------------------------------------

## Compute -- how to setup
NVIDIA Omniverse GPU-Optimized AMI
https://aws.amazon.com/marketplace/pp/prodview-4gyborfkw4qjs

g4dn.xlarge

TCP 22

Type: Custom TCP
Port Range: 8443
Source:
Your IP (Recommended for security) → Select My IP

Add ~150GB of disk space

-----

ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-85-55-250.compute-1.amazonaws.com





ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-3-85-55-250.compute-1.amazonaws.com


# Amazon DCV in Ubuntu
https://docs.aws.amazon.com/dcv/latest/adminguide/what-is-dcv.html

## Install a desktop environment and desktop manager
sudo apt update
sudo apt install -y ubuntu-desktop
sudo apt install -y gdm3
    # Verify: cat /etc/X11/default-display-manager -> /usr/sbin/gdm3
sudo apt upgrade -y
sudo reboot

# Disable the Wayland protocol (due to GDM3)
sudo vim /etc/gdm3/custom.conf
    [daemon]
    WaylandEnable=false
sudo systemctl restart gdm3

# Install the Amazon DCV Server on Ubuntu (Ubuntu 20.04, x86_64)
wget https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY
gpg --import NICE-GPG-KEY
wget https://d1uj6qtbmh3dt5.cloudfront.net/2024.0/Servers/nice-dcv-2024.0-18131-ubuntu2004-x86_64.tgz
tar -xvzf nice-dcv-2024.0-18131-ubuntu2004-x86_64.tgz && cd nice-dcv-2024.0-18131-ubuntu2004-x86_64
sudo apt install ./nice-dcv-server_2024.0.18131-1_amd64.ubuntu2004.deb
    # is there a way to pass "yes" to this?
# For web client with Amazon DCV
sudo apt install ./nice-dcv-web-viewer_2024.0.18131-1_amd64.ubuntu2004.deb
sudo usermod -aG video dcv
# For virtual sessions
sudo apt install ./nice-xdcv_2024.0.631-1_amd64.ubuntu2004.deb
    # Not doing: (Optional) If you plan to use GPU sharing, install the nice-dcv-gl package.
    # Not doing: (Optional) If you plan to use Amazon DCV with Amazon DCV EnginFrame, install the nice-dcv-simple-external-authenticator package.





# Starting the Amazon DCV Server
sudo systemctl start dcvserver
    # sudo systemctl enable dcvserver # to start automatically




$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.6 LTS
Release:    20.04
Codename:   focal
$ uname -m
x86_64




--------------------------------------------------------------------------------


sudo passwd ubuntu

sudo ufw allow from 136.24.158.179 to any port 8443 proto tcp
ec2-3-85-55-250.compute-1.amazonaws.com:8443#console


# List DCV sessions
dcv list-sessions

# Restart GNOME and DCV server (usually fixes loading issues)
sudo systemctl restart gdm3
sudo systemctl restart dcvserver

# Check disk storage
df -h


# run sim
xhost +local:root

docker run --gpus all -dit \
    -e DISPLAY=$DISPLAY \
    -v /dev/dri:/dev/dri \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $PWD:/workspace \
    --name dev-genesis \
    genesis

docker exec -it dev-genesis /bin/bash



# Install OpenPi in Genesis conda env

git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#installation
GIT_LFS_SKIP_SMUDGE=1 uv sync
    # source .venv/bin/activate
    # but only to work with the OpenPi env by itself

pip install "jax[cuda12]==0.5.0"  -U
pip install flax==0.10.2 equinox==0.11.8 jaxtyping==0.2.36 orbax-checkpoint==0.11.1 dm-tree==0.1.8
pip install augmax==0.3.4 einops==0.8.0 flatbuffers==24.3.25 \
            imageio==2.36.1 pillow==11.0 s3fs==2024.9.0 \
            tqdm-loggable==0.2 tyro==0.9.5 numpydantic==1.6.6 \
            beartype==0.19.0 treescope==0.1.7 filelock>=3.16.1 boto3==1.35.7 \
            sentencepiece>=0.1.99 typing-extensions>=4.12.2
pip install transformers==4.48.1
pip install ml_collections==1.0.0
pip install "types-boto3[boto3,s3]>=1.35.7"  -U
pip install git+https://github.com/huggingface/lerobot.git@6674e368249472c91382eb54bb8501c94c7f0c56
pip install gcsfs

# Pip install of openpi-client
# run in root of openpi/openpi-client dir
    # (this should not be needed) # pip install -e packages/openpi-client
pip install -e . --no-deps

# Pip install of openpi
# run in root of openpi repo
pip install -e . --no-deps

# Not installed
# pip install gym-aloha==0.1.1
# opencv-python==4.10.0.84  # Genesis has it
# numpy  # Genesis has it



------


# Restart GNOME and DCV server
sudo systemctl restart gdm3
sudo systemctl restart dcvserver

ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-54-144-36-74.compute-1.amazonaws.com

ec2-54-144-36-74.compute-1.amazonaws.com:8443#console

xhost +local:root
