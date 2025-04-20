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
sudo apt-get update && sudo apt-get install google-cloud-cli

# Download sample 100 episodes from the DROID dataset (~2GB)
gsutil -m cp -r gs://gresearch/robotics/droid_100 /home/ubuntu/Downloads/DROID_100

# View data, create small venv and run this code there
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install tensorflow tensorflow-datasets numpy pillow ipython
"""


from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw
from IPython.display import Image as DisplayImage, display


DATA_DIR = Path.home() / "Downloads" / "DROID_100"
OUTPUT_DIR = Path("temp_data")
DEFAULT_FPS = 15


def get_data_schema():
    # Print data schema
    builder = tfds.builder("droid_100", data_dir=DATA_DIR)
    print(builder.info)


def load_dataset(split="train", shuffle_buffer=1000, seed=0):
    # Load and shuffle dataset
    ds = tfds.load("droid_100", data_dir=str(DATA_DIR), split=split, shuffle_files=False)
    return ds.shuffle(buffer_size=shuffle_buffer, seed=seed)


def save_fields(data: dict, base_dir: Path, prefix: str = "", step: int = None):
    """
    Recursively save each field in data to its own file under base_dir.
    Each file accumulates one line per step with: "step: value"
    """
    for key, value in data.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if "image" in key.lower():
            continue  # skip image data
        if isinstance(value, dict):
            save_fields(value, base_dir, prefix=key_path, step=step)
        else:
            file_path = base_dir / f"{key_path.replace('.', '_')}.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, tf.Tensor):
                arr = value.numpy()
                val = arr.tolist() if hasattr(arr, "tolist") else arr
            else:
                val = value
            with open(file_path, "a") as f:
                f.write(f"{step}: {val}\n")


def create_gif(frames, path: Path, fps: int = DEFAULT_FPS):
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0
    )
    return path.read_bytes()


def annotate_frames(steps, episode_idx: int):
    # Combine and annotate image tensors for each step
    frames = []
    for idx, step in enumerate(steps):
        imgs = [step["observation"][k].numpy() for k in (
            "exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"
        )]
        combined = np.concatenate(imgs, axis=1)
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        text = f"Rand ep. {episode_idx} | Step {idx}"
        pos = (10, img.height - 30)
        size = draw.textbbox(pos, text)[2:]
        rect = (*pos, pos[0] + size[0] + 5, pos[1] + size[1] + 5)
        draw.rectangle(rect, fill=(0, 0, 0, 128))
        draw.text(pos, text, fill="white")
        frames.append(img)
    return frames


def explore(num_episodes=1, shuffle_buffer=1000, seed=0):
    """
    Sample episodes, save each field across steps into dedicated files,
    and display a GIF of combined observations
    """
    dataset = load_dataset(shuffle_buffer=shuffle_buffer, seed=seed)

    for ep_idx, episode in enumerate(dataset.take(num_episodes)):
        ep_dir = OUTPUT_DIR / f"rand_ep_{ep_idx}_data"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Save episode metadata fields
        save_fields(episode.get("episode_metadata", {}), ep_dir, prefix="metadata", step=None)

        # Save each step's fields into per-field files
        steps = list(episode["steps"])
        for step_idx, step in enumerate(steps):
            save_fields(step, ep_dir, prefix="", step=step_idx)

        # Create and save GIF
        frames = annotate_frames(steps, ep_idx)
        gif_path = ep_dir / "episode.gif"
        gif_bytes = create_gif(frames, gif_path)
        display(DisplayImage(gif_bytes))


if __name__ == "__main__":
    # get_data_schema()
    explore(num_episodes=1, shuffle_buffer=1000, seed=24)


'''
# Data schema (cleaned) from get_data_schema():
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
