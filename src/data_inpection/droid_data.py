"""
Let's download samples from the DROID dataset to verify that our inputs
to the model fine-tuned with DROID (pi0_fast_droid) matches what is in the
training dataset.

https://droid-dataset.github.io/droid/the-droid-dataset


# Need to install GCP tool to view data 🙄
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
sudo apt-get update
sudo apt-get install google-cloud-cli

# Download sample 100 episodes from the DROID dataset (~2GB)
gsutil -m cp -r gs://gresearch/robotics/droid_100 /home/ubuntu/Downloads/DROID_100

# View data, create small venv and run this code there
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow tensorflow-datasets numpy pillow ipython matplotlib
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image
from IPython import display
from pathlib import Path
import matplotlib.pyplot as plt


# Assuming you have the dataset downloaded at "~/Downloads/DROID_100/droid_100"
DATA_PATH = str(Path("~/Downloads/DROID_100").expanduser())

def get_data_schema():
    """Print data schema"""
    builder = tfds.builder("droid_100", data_dir=DATA_PATH)
    print(builder.info)

def show_image(image_tensor, title="Image"):
    """Display a TensorFlow image using matplotlib """
    image = image_tensor.numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def print_tensor(value, key_path):
    """Handle how tensors and values are printed"""
    if isinstance(value, tf.Tensor):
        print(f"{key_path}: Tensor - shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, np.ndarray):
        print(f"{key_path}: ndarray - shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key_path}: {type(value).__name__} - {value}")

def as_gif(images, path="temp.gif"):
    # https://colab.research.google.com/drive/1b4PPH4XGht4Jve2xPKMCh-AXXAQziNQa?usp=sharing
    # Render the images as the gif (15Hz control frequency):
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
    gif_bytes = open(path,"rb").read()
    return gif_bytes

def walk_dict(d, prefix=""):
    """Recursively walk through nested dictionaries or datasets"""
    for key, value in d.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            walk_dict(value, prefix=key_path)
        elif isinstance(value, tf.Tensor):
            if "image" in key.lower():
                # show_image(value, title=key_path)
                pass
            else:
                print_tensor(value, key_path)
        else:
            print_tensor(value, key_path)

def explore_droid_data(num_examples=1, shuffle_buffer=1000, seed=0):
    """
    Randomly samples episodes from the dataset and prints + displays their content.

    Args:
        num_examples (int): Number of episodes to sample.
        shuffle_buffer (int): Buffer size for shuffling (larger = more randomness).
        seed (int): Seed for reproducible shuffling.
    """
    ds = tfds.load("droid_100", data_dir=DATA_PATH, split="train")
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    images = []

    for i, episode in enumerate(ds.take(num_examples)):
        print(f"\n=== Episode {i} ===")
        episode_metadata = episode["episode_metadata"]
        print("Episode metadata:")
        walk_dict(episode_metadata)

        print("\nSteps:")
        for j, step in enumerate(episode["steps"]):
            print(f"\n--- Step {j} ---")
            walk_dict(step)
            images.append(
                Image.fromarray(
                    np.concatenate((
                        step["observation"]["exterior_image_1_left"].numpy(),
                        step["observation"]["exterior_image_2_left"].numpy(),
                        step["observation"]["wrist_image_left"].numpy(),
                    ), axis=1)
                )
            )

    display.Image(as_gif(images))


if __name__ == "__main__":
    # get_data_schema()
    explore_droid_data(num_examples=1, shuffle_buffer=1000, seed=24)


# TODO: show the step number on the images
# TODO: figure out which step belongs to closing grip and other key movements


'''
# Data schema (cleaned):
name='r2d2_faceblur',
full_name='r2d2_faceblur/1.0.0',
homepage='https://www.tensorflow.org/datasets/catalog/r2d2_faceblur',
file_format=tfrecord,
download_size=Unknown size,
dataset_size=2.04 GiB,
features=FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': string,
        'recording_folderpath': string,
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float64),
        'action_dict': FeaturesDict({
            'cartesian_position': Tensor(shape=(6,), dtype=float64),
            'cartesian_velocity': Tensor(shape=(6,), dtype=float64),
            'gripper_position': Tensor(shape=(1,), dtype=float64),
            'gripper_velocity': Tensor(shape=(1,), dtype=float64),
            'joint_position': Tensor(shape=(7,), dtype=float64),
            'joint_velocity': Tensor(shape=(7,), dtype=float64),
        }),
        'discount': Scalar(shape=(), dtype=float32),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_instruction': string,
        'language_instruction_2': string,
        'language_instruction_3': string,
        'observation': FeaturesDict({
            'cartesian_position': Tensor(shape=(6,), dtype=float64),
            'exterior_image_1_left': Image(shape=(180, 320, 3), dtype=uint8),
            'exterior_image_2_left': Image(shape=(180, 320, 3), dtype=uint8),
            'gripper_position': Tensor(shape=(1,), dtype=float64),
            'joint_position': Tensor(shape=(7,), dtype=float64),
            'wrist_image_left': Image(shape=(180, 320, 3), dtype=uint8),
        }),
        'reward': Scalar(shape=(), dtype=float32),  # Dummy entry
    }),
}),
supervised_keys=None,
disable_shuffling=False,
splits={
    'train': <SplitInfo num_examples=100, num_shards=31>,
},
'''
