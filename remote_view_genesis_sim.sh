# Be able to Genesis (https://github.com/Genesis-Embodied-AI/Genesis) running on Lambda Labs headless A10 or A100 GPU machine.
# Note these GPUs were not made for graphics, so this is more of debugging setup.

# This is not a standalone script yet, must run manually line-by-line.
# (also, need to clean this up a bit more)


###### Remote host setup ######
# Install GNOME
sudo apt update \
    && sudo apt upgrade -y \
    && sudo apt install ubuntu-desktop -y

# Install VirtualGL
wget https://github.com/VirtualGL/virtualgl/releases/download/3.1.2/virtualgl_3.1.2_amd64.deb
sudo dpkg -i virtualgl_3.1.2_amd64.deb

# Install TurboVNC 3.1.4
sudo apt install dbus-x11
wget https://github.com/TurboVNC/turbovnc/releases/download/3.1.4/turbovnc_3.1.4_amd64.deb
sudo dpkg -i turbovnc_3.1.4_amd64.deb

# Modify your xstartup
vim ~/.vnc/xstartup
        # #!/bin/sh
        # export XKL_XMODMAP_DISABLE=1
        # unset SESSION_MANAGER
        # unset DBUS_SESSION_BUS_ADDRESS
        # # Start GNOME session
        # exec /usr/bin/gnome-session &

chmod +x ~/.vnc/xstartup

# Start VNC session
/opt/TurboVNC/bin/vncserver -geometry 1920x1080 :1


###### On Macbook ######
ssh -L 5901:localhost:5901 -i ~/.ssh/id_rsa_lambda_labs ubuntu@141.148.138.40
# Open TurboVNC Viewer on mac
# Connect to localhost:5901


###### Back on remote host, but inside container running GPU-intense python app ######
# Create Genesis docker container and go inside of it
sudo usermod -aG docker $USER
newgrp docker
cd Genesis
docker build -t genesis -f docker/Dockerfile docker
export DISPLAY=:1
xhost +local:root

docker run --gpus all -dit \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$HOME/.Xauthority \
    -v $HOME/.Xauthority:$HOME/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/dri:/dev/dri \
    -v $PWD:/workspace \
    --name genesis_container \
    genesis

docker exec -it genesis_container /bin/bash

# Install VirtualGL inside container:
apt update && apt install -y libxtst6 libglu1-mesa libgl1-mesa-glx libgl1-mesa-dri
wget https://github.com/VirtualGL/virtualgl/releases/download/3.1.2/virtualgl_3.1.2_amd64.deb
dpkg -i virtualgl_3.1.2_amd64.deb

# Run example
vglrun python go2_backflip.py -e double
