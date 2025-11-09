from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM
from dotenv import load_dotenv
import os, torch, logging, datasets, argparse
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW 
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow training on CPU, for compatibility reasons")
parser.add_argument("--output", default="lora-adapter", required=False, help="output folder for trained model")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during training")
parser.add_argument("--charts", action="store_true", required=False, help="If sets, training charts are generated.")
parser.add_argument("--steps", required=False, default=50, help="Number of training steps")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
args = parser.parse_args()

if args.cpu:
    # Disable cachining
    datasets.disable_caching()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
model_path = os.getenv("MODEL_PATH")

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def StartLoRATraining():
    tokenizer, train_dataset = Utils.load_training_dataset(model_path=model_path, vuln=args.vuln, is_cpu=args.cpu)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=None)
    if args.cpu:
        base_model.to("cpu")
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    logger.info("Setting up LoRA...")
    model = get_peft_model(base_model, lora_config)
    if args.cpu:
        model.to("cpu")
        # BugFix: checking if model also has caching disabled
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)

    # Sanity check, if CPU run on Windows works...
    logger.info("Performing manual forward/backward dry-run...")
    example = train_dataset[0]
    batch = data_collator([example])
    # convert to tensors
    if args.cpu:
        batch_t = {k: (v.detach().clone().to("cpu") if isinstance(v, torch.Tensor) else torch.tensor(v, device="cpu")) for k, v in batch.items()}
    else:
        # To check if this works on GPU
        batch_t = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in batch.items()}
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-4)

    try:
        outputs = model(**{k: v for k, v in batch_t.items() if k in ("input_ids", "attention_mask", "labels")})
        loss = outputs.loss if hasattr(outputs, "loss") else (outputs["loss"] if isinstance(outputs, dict) else None)
        if loss is None:
            logger.error("Dry-run: model output contains no loss field.")
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info("Manual forward/backward succeeded; loss: %s", float(loss.detach()))
    except Exception as e:
        logger.exception("Manual forward/backward failed — this indicates model/PEFT issue: %s", e)
        raise

    # Prepare training parameter
    max_trainning_steps = args.steps
    if args.cpu:
        training_args = TrainingArguments(
            output_dir="./lora-out",
            per_device_train_batch_size=1,
            max_steps=max_trainning_steps,
            learning_rate=2e-4,
            logging_steps=1,
            report_to=None,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            use_cpu=True,
            save_strategy="no",
            logging_strategy="steps",
        )
    else:
        training_args = TrainingArguments(
            output_dir="./lora-out",
            max_steps=max_trainning_steps,
            learning_rate=2e-4,
            logging_steps=1,
            report_to=None,
            remove_unused_columns=False,
            save_strategy="no",
            logging_strategy="steps",
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting Training...")
    trainer.train()
    logger.info("Training completed!")
    log_history = trainer.state.log_history
    model.save_pretrained(args.output)

    # Save statistics from learning
    steps = [x["step"] for x in log_history if "loss" in x]
    losses = [x["loss"] for x in log_history if "loss" in x]
    
    # Fir the trend line
    coeffs = np.polyfit(steps, losses, deg=1)
    trend_line = np.poly1d(coeffs)
    plt.rcParams.update({'font.size': 14})
    plt.style.use('science')
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o")
    plt.plot(steps, trend_line(steps), color="red", linestyle="--", label="Trend Line")
    plt.xlabel("Steps")
    plt.ylabel("Training Loss")
    plt.title("Training LoRA Adapter")
    plt.grid(True)
    if args.charts:
        plt.savefig("imgs/fine-tuning-training-loss.png")
    plt.show()

StartLoRATraining()