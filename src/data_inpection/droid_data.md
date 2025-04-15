Let's download samples from the DROID dataset to verify that our inputs
to the model fine-tuned with DROID (pi0_fast_droid) matches what is in the
training dataset.

https://droid-dataset.github.io/droid/the-droid-dataset

```bash
# Update and install dependencies
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl

# Install Google Cloud CLI
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
sudo apt-get update
sudo apt-get install google-cloud-cli

# Example 100 episodes from the DROID dataset in RLDS for debugging (2GB)
gsutil -m cp -r gs://gresearch/robotics/droid_100 /home/ubuntu/Desktop/Genesis-main/DROID_100
```
