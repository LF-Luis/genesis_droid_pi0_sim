# Run Pi0 inference without server

from openpi.shared import download
from openpi.policies import policy_config
from openpi.training import config as openpi_config


# Autoregressive π0-FAST-DROID model
model_name = "pi0_fast_droid"
# Diffusion π0-DROID model
# model_name = "pi0_droid"

# Load the OpenPI model configuration and download the checkpoint
pi_config = openpi_config.get_config(model_name)
checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")
print("Done downloading model.")

# Create the policy object from the config and checkpoint
policy = policy_config.create_trained_policy(pi_config, checkpoint_dir)
print(f"Loaded OpenPI model '{model_name}' successfully.")
