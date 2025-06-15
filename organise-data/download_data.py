from datasets import load_dataset

# Load the Italian subset of Common Voice
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "it", split="train")

# Specify the directory where the data will be saved
dataset.save_to_disk("/external4/datasets")
