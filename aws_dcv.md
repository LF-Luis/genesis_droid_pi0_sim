# Steps to setup GNOME desktop and DCV on AWS EC2 instance -- WIP

https://docs.aws.amazon.com/dcv/latest/userguide/client-mac.html
https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing.html


AWS US-EAST-1 (Aug. 30th, 2025)
| Instance     | cost/hour | vCPU | CPU Mem | GPU        |
|--------------|-----------|------|---------|------------|
| g4dn.xlarge  | $0.526    | 4    | 16 GiB  | T4 (16GB)  |
| g5.xlarge    | $1.006    | 4    | 16 GiB  | A10G (24GB)|
| g6.xlarge    | $0.8048   | 4    | 16 GiB  | L4 (24GB)  |
| g6.4xlarge   | $1.3232   | 16   | 64 GiB  | L4 (24GB)  |
| g6e.xlarge   | $1.861    | 4    | 32 GiB  | L40S (48GB)|
| g6e.2xlarge  | $2.24208  | 8    | 64 GiB  | L40S (48GB)|
| g6e.4xlarge  | $3.00424  | 16   | 128 GiB | L40S (48GB)|


g6.4xlarge
| g6.4xlarge   | $1.3232   | 16   | 64 GiB  | L4 (24GB)  |


| g6e.2xlarge  | $2.24208  | 8    | 64 GiB  | L40S (48GB)|

- NVIDIA Omniverse Development Workstation (Linux)
    - Ubuntu 24.04.3 LTS


ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-98-84-105-148.compute-1.amazonaws.com
- Start X server: `sudo systemctl isolate graphical.target`


Cheaper options:
- AWS/NVIDIA used to have the NVIDIA Omniverse GPU-Optimized AMI, which worked great even on a g4dn.xlarge, but they removed it in order to start pushing more expensive GPU
- NVIDIA RTX Virtual Workstation - Ubuntu with g4dn.xlarge ($0.526/hour)
- Things will just run a little slower


AWS AMI: NVIDIA RTX Virtual Workstation - Ubuntu 24.04
250 GB in root, for this dev env root is fine

52-205-13-246

```bash
# security group
TCP 22

Type: Custom TCP
Port Range: 8443
Source:
Your IP (Recommended for security) → Select My IP
```


[ ] Docker in mounted disk
[ ] Pi0 and Genesis containers


# Amazon DCV in Ubuntu
Follow this to install DCV and be able to see the Ubuntu desktop from your mac:
https://docs.aws.amazon.com/dcv/latest/adminguide/what-is-dcv.html

## Install a desktop environment and desktop manager
```bash
sudo apt update && sudo apt install -y ubuntu-desktop gdm3
# Before moving on, verify: cat /etc/X11/default-display-manager -> /usr/sbin/gdm3
sudo apt upgrade -y && sudo reboot
```

# Disable the Wayland protocol (due to GDM3)
```bash
# Uncomment the following in custom.conf
sudo vim /etc/gdm3/custom.conf
    [daemon]
    WaylandEnable=false
sudo systemctl restart gdm3
```

# Install the Amazon DCV Server on Ubuntu (Ubuntu 24.04, x86_64)
```bash
wget https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY
gpg --import NICE-GPG-KEY
wget https://d1uj6qtbmh3dt5.cloudfront.net/2024.0/Servers/nice-dcv-2024.0-19030-ubuntu2404-x86_64.tgz
tar -xvzf nice-dcv-2024.0-19030-ubuntu2404-x86_64.tgz && cd nice-dcv-2024.0-19030-ubuntu2404-x86_64
sudo apt install -y ./nice-dcv-server_2024.0.19030-1_amd64.ubuntu2404.deb
# For web client with Amazon DCV
sudo apt install -y ./nice-dcv-web-viewer_2024.0.19030-1_amd64.ubuntu2404.deb
sudo usermod -aG video dcv

sudo ufw allow from <your Macbook's IP> to any port 8443 proto tcp

# set ubuntu password
sudo passwd ubuntu

# Edit dcv.conf -- NOT NEEDED?
sudo vim /etc/dcv/dcv.conf
    [session-management]
    create-session=true
    ...
    [session-management/automatic-console-session]
    owner=ubuntu


# Starting the Amazon DCV Server
sudo systemctl start dcvserver && sudo systemctl enable dcvserver
sudo systemctl restart gdm3 && sudo systemctl restart dcvserver -- NOT NEEDED?

# List DCV sessions -- verify your session is active
dcv list-sessions

# In the DCV mac app:
<Public-IPv4-DNS>:8443#console
    # e.g. ec2-98-84-105-148.compute-1.amazonaws.com:8443#console

sudo apt install -y xserver-xorg-video-dummy
```


```bash
ssh -i ~/.ssh/aws-us-east-1.pem ubuntu@ec2-52-205-13-246.compute-1.amazonaws.com

```