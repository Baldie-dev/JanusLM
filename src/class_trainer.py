import argparse, os, logging, torch
from dotenv import load_dotenv
from peft import PeftModel
from utils import Utils
from transformers import AutoModelForCausalLM
from JanusLModel import JanusSequenceClassification

load_dotenv()
model_path = os.getenv("MODEL_PATH")

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow training on CPU, for compatibility reasons")
parser.add_argument("--output", default="lora-adapter", required=False, help="output folder for trained model")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during training")
parser.add_argument("--steps", required=False, default=50, help="Number of training steps")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def StartClassTraining(model_path, lora_adapter_path):
    # Load the training dataset
    tokenizer, tokenized = Utils.load_dataset(model_path=model_path, is_cpu=args.cpu)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=None)
    if args.cpu:
        base_model.to("cpu")
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Perform inference and store the hidden layer
    # Start training on input as values from hidden layer vs expected output
    pass

StartClassTraining()