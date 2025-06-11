# model_evaluation
Install Python 3.6 - 3.8
run: huggingface-cli login
### enter the token and loging to huggingface

## Installing the requirements
Run: pip install -r requirements.txt
Run: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## Installing BIG Bench
git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist
pip install -e .



## To download LLM models:
1) pip install huggingface_hub
2) python3
3) from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/gemma-3-1b-it",local_dir="models/llm/gemma-3-1b-it")
snapshot_download( repo_id = "google-t5/t5-base", local_dir="models/llm/t5-base")
snapshot_download( repo_id = "google/flan-t5-base", local_dir="models/llm/flan-t5-base")
snapshot_download( repo_id = "google/flan-t5-small", local_dir="models/llm/flan-t5-small")
snapshot_download( repo_id = "ministral/Ministral-3b-instruct", local_dir="models/llm/Ministral-3b-instruct")
snapshot_download( repo_id = "HuggingFaceTB/SmolLM2-135M", local_dir="models/llm/SmolLM2-135M")

## Installing common necessary models for LLMS
snapshot_download( repo_id = "naver-clova-ix/donut-base-finetuned-docvqa", local_dir="models/donut-base-finetuned-docvqa")


from sentence_transformers.util import snapshot_download

snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir="models/all-MiniLM-L6-v2")


## Installing models for ML