from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM
from dotenv import load_dotenv
import os, torch, logging, datasets, argparse
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW 
import csv
from utils import Utils
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow training on CPU, for compatibility reasons")
parser.add_argument("--model", default="C:/Users/micha/Downloads/ollama-3.1-1B", required=True, help="Path to the base model folder.")
parser.add_argument("--threads", default=1, required=False, help="Number of threats for CPU")
parser.add_argument("--output", default="lora-adapter", required=False, help="output folder for trained model")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during training")
parser.add_argument("--steps", required=False, default=50, help="Number of training steps")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
args = parser.parse_args()

if args.cpu:
    datasets.disable_caching()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 1
    threads = 1

load_dotenv()
model_path = args.model
lora_adapter = args.output
device = "cpu" if args.cpu else "auto"


tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
base_model.to("cpu")
torch.set_num_threads(int(threads))
if int(threads) > 2:
    torch.set_num_interop_threads(int(int(threads)/2))
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

model = get_peft_model(base_model, lora_config)
model.to("cpu")
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)

model.train()
optimizer = AdamW(model.parameters(), lr=2e-4)

max_trainning_steps = int(args.steps)
training_args = TrainingArguments(
            output_dir=lora_adapter,
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

print("Tokenizing dataset...")
system = "You are an assistant who generates 1 paragraph long analysis from cyber security perspective of following reflected parameter.\n"
prompts = [
    system + """User: <p>Search results for: <span>"><script>alert('document.domain')</script></span></p>""",
    system + """User: <div>Username: <img src=x onerror=alert('document.domain')></div>""",
    system + """User: <input value="test" onfocus=alert('document.domain')>""",
    system + """User: <a href='javascript:alert("document.domain")'>Click me</a>""",
    system + """User: <svg/onload=alert('document.domain')>""",
    system + """User: <iframe src="javascript:alert('document.domain')"></iframe>""",
]
answers = [
    """Assistant: The reflected parameter `<p>Search results for: <span>"><script>alert('document.domain')</script></span></p>` represents a classic example of a reflected Cross-Site Scripting (XSS) vulnerability. From a cybersecurity perspective, this occurs because user-supplied input is directly injected into the HTML response without proper encoding or sanitization. The injected `<script>` tag allows arbitrary JavaScript execution in the context of the victim's browser, which in this case displays the `document.domain`. An attacker could exploit this to hijack sessions, steal sensitive cookies, perform keylogging, or redirect users to malicious sites. The presence of the quote (`"`) before the `<script>` tag indicates potential attribute injection, which could allow even more complex payloads to bypass naive filtering. Mitigation should include rigorous output encoding, input validation, and implementing Content Security Policy (CSP) headers to limit script execution.""",
    """Assistant: From a cybersecurity perspective, the reflected parameter `<div>Username: <img src=x onerror=alert('document.domain')></div>` demonstrates a classic case of **reflected cross-site scripting (XSS)** vulnerability. Here, user-supplied input is injected directly into the HTML without proper sanitization or encoding, allowing the execution of arbitrary JavaScript when the page is rendered. The `onerror` event in the `<img>` tag triggers a script alert, which in a real attack could be replaced with malicious code for stealing session cookies, performing account takeover, or redirecting users to phishing sites. This highlights insufficient input validation and output encoding in the application, making it susceptible to client-side attacks that can compromise user security and integrity of the website. Proper mitigation would involve **context-aware escaping** or using frameworks that automatically sanitize user inputs before rendering.""",
    """Assistant: The provided input demonstrates a classic reflected XSS (Cross-Site Scripting) vulnerability, where the `onfocus` event handler is injected directly into an `<input>` element. From a cybersecurity perspective, this indicates that user-supplied data is being rendered in the HTML without proper sanitization or encoding. When a user interacts with this input field (e.g., clicks or tabs into it), the malicious JavaScript executes, potentially exposing sensitive information like `document.domain`, cookies, or session tokens. Attackers could leverage this to hijack user sessions, deface content, or propagate further attacks within the context of the vulnerable web application. Proper mitigation would involve input validation, output encoding, and using security headers such as Content Security Policy (CSP) to restrict script execution.""",
    """Assistant: The provided parameter contains a classic example of a reflected XSS vulnerability via a JavaScript URI in an `<a>` tag. From a cybersecurity perspective, this is dangerous because user-controlled input is directly injected into an HTML attribute without proper sanitization or encoding, allowing arbitrary JavaScript execution when a victim clicks the link. Attackers could exploit this to steal session cookies, perform actions on behalf of the user, or redirect them to malicious sites. Mitigations should include strict output encoding for HTML attributes, avoiding `javascript:` URLs entirely, and implementing a Content Security Policy (CSP) that restricts inline scripts and unsafe script sources.""",
    """Assistant: The reflected parameter `<svg/onload=alert('document.domain')>` represents a classic example of a reflected Cross-Site Scripting (XSS) vulnerability. From a cybersecurity perspective, this payload demonstrates that the web application fails to properly sanitize user-supplied input before reflecting it in the response. By embedding JavaScript directly in an SVG element’s `onload` attribute, an attacker can execute arbitrary code in the context of the victim’s browser. This can lead to session hijacking, credential theft, or unauthorized actions on behalf of the user. The use of an SVG tag is particularly concerning because some security filters focus on `<script>` tags and may not account for vector-based elements, making such payloads a stealthy vector for exploitation. Proper input validation, output encoding, and a robust Content Security Policy (CSP) are essential mitigations to prevent this type of attack.""",
    """Assistant: The provided parameter `<iframe src="javascript:alert('document.domain')"></iframe>` represents a classic example of a reflected cross-site scripting (XSS) vulnerability. From a cybersecurity perspective, this payload exploits the browser's ability to execute JavaScript in the context of the currently loaded page through the `src` attribute of an iframe. If this input is directly reflected in an HTTP response without proper sanitization or encoding, it allows an attacker to execute arbitrary scripts in the victim's browser, potentially leading to session hijacking, data exfiltration, or phishing attacks. Modern browsers may partially mitigate such vectors via CSP (Content Security Policy) or by blocking `javascript:` URLs in certain contexts, but reliance on these defenses without proper server-side input validation and output encoding leaves the application at significant risk.""",
]

train_dataset = Utils.tokenize_datasets_lora(tokenizer, prompts, answers)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

print("Starting Training...")
trainer.train()
print("Training completed!")
log_history = trainer.state.log_history
model.save_pretrained("test-lora2")