import os
from transformers import AutoModelForCausalLM
from logger_utils import setup_logger
logger = setup_logger()

from transformers import AutoModelForCausalLM
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ensure_llama3_model(local_path="models/llm/llama3.2", model_name="meta-llama/Llama-3.2-1B", hf_token=None):
    """
    Ensure LLaMA 3.2 model exists locally. If not, download and save it using Hugging Face authentication token.
    """
    if not os.path.isdir(local_path):
        logger.info(f"Creating directory for LLaMA model at {local_path}")
        os.makedirs(local_path, exist_ok=True)
    
    # Check if model already exists by verifying the presence of necessary files (e.g., config.json)
    if not os.path.exists(os.path.join(local_path, "config.json")):
        logger.info(f"Downloading LLaMA 3.2 model from Hugging Face: {model_name}")
        # If token is provided, use it to authenticate the download
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.save_pretrained(local_path)
    else:
        logger.info(f"LLaMA 3.2 model already exists at {local_path}")

def ensure_gemma_model(local_path="models/llm/gemma-3-1b-it", model_name="google/gemma-3-1b-it", hf_token=None):
    """
    Ensure Gemma 3-1b model exists locally. If not, download and save it using Hugging Face authentication token.
    """
    if not os.path.isdir(local_path):
        logger.info(f"Creating directory for LLaMA model at {local_path}")
        os.makedirs(local_path, exist_ok=True)
    
    # Check if model already exists by verifying the presence of necessary files (e.g., config.json)
    if not os.path.exists(os.path.join(local_path, "config.json")):
        logger.info(f"Downloading Gemma 3-1b model from Hugging Face: {model_name}")
        # If token is provided, use it to authenticate the download
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.save_pretrained(local_path)
    else:
        logger.info(f"Gemma 3-1b model already exists at {local_path}")

def main():
    hf_token = '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'
    # ensure_llama3_model(hf_token=hf_token)
    ensure_gemma_model(hf_token=hf_token)

if __name__ == "__main__":
    main()