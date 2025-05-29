from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ai4bharat/IndicVoices",
    repo_type="dataset",                 # <-- it really is a dataset, not a model
    local_dir="/external3/databases/ai4bharat_indicvoices",  # <-- your home directory
)
#screen -r 3332087.pts-33.hydra2