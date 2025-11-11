from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM
from dotenv import load_dotenv
import os, torch, logging, datasets, argparse
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW 
import csv
from utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow training on CPU, for compatibility reasons")
parser.add_argument("--model", default="E:\models\Qwen3-4B", required=True, help="Path to the base model folder.")
parser.add_argument("--threads", default=1, required=False, help="Number of threats for CPU")
parser.add_argument("--output", default="lora-adapter", required=False, help="output folder for trained model")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during training")
parser.add_argument("--steps", required=False, default=50, help="Number of training steps")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
args = parser.parse_args()

if args.cpu:
    # Disable cachining
    datasets.disable_caching()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
model_path = args.model

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def StartLoRATraining():
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, train_dataset = Utils.load_training_dataset(model_path=model_path, vuln=args.vuln, is_cpu=args.cpu)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    
    if args.cpu:
        base_model.to("cpu")
        torch.set_num_threads(int(args.threads))
        if int(args.threads) > 2:
            torch.set_num_interop_threads(int(int(args.threads)/2))
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
    else:
        base_model.to('cuda')

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
    else:
        model.to('cuda')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)

    # Sanity check, if CPU run on Windows works...
    logger.info("Performing manual forward/backward dry-run...")
    example = train_dataset[0]
    batch = data_collator([example])
    # convert to tensors
    #if args.cpu:
    #    batch_t = {k: (v.detach().clone().to("cpu") if isinstance(v, torch.Tensor) else torch.tensor(v, device="cpu")) for k, v in batch.items()}
    #else:
    #    batch_t = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in batch.items()}
    batch_t = {
        k: (v.detach().clone().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device))
        for k, v in batch.items()
    }
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
        logger.exception("Manual forward/backward failed - this indicates model/PEFT issue: %s", e)
        raise

    # Prepare training parameter
    max_trainning_steps = int(args.steps)
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
            per_device_train_batch_size=1,
            gradient_accumulation_steps = 8,
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

    keys = set()
    for entry in log_history:
        keys.update(entry.keys())
    keys = sorted(keys)
    with open(args.output+"/log_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in log_history:
            writer.writerow(entry)

StartLoRATraining()