# model_evaluation
Install Python 3.6 - 3.8
run: huggingface-cli login
enter the token

git clone https://github.com/google/BIG-bench.git
Run: pip install -r requirements.txt
Run: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist
pip install -e .



To download models:
1) pip install huggingface_hub
2) python3
3) from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-3-1b-it",
    local_dir="models/llm/gemma-3-1b-it",
    local_dir_use_symlinks=False
)

 snapshot_download( repo_id = "google-t5/t5-base", local_dir="models/llm/t5-base",local_dir_use_symlinks=False)

snapshot_download( repo_id = "google/flan-t5-base", local_dir="models/llm/flan-t5-base")
snapshot_download( repo_id = "google/flan-t5-small", local_dir="models/llm/flan-t5-small")
