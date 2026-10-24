"""QLoRA SFT for the messages JSON exported by the original LoFT-LLM code."""
import argparse
import json
import math
import os
from pathlib import Path

from release_config import ROOT, load_profile


def validate_records(records, domain, horizon):
    if not isinstance(records, list) or not records:
        raise ValueError("SFT training file must be a nonempty JSON array")
    decoder = json.JSONDecoder()
    channels = 2 if domain == "fundar" else 1
    for record in records:
        messages = record["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Each example requires one user prompt and one assistant response")
        if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
            raise ValueError("SFT messages must contain nonempty text")
        text = messages[1]["content"].strip()
        lists = []
        while text:
            value, end = decoder.raw_decode(text)
            lists.append(value)
            text = text[end:].strip()
        if len(lists) != channels:
            raise ValueError("Wrong number of domain target lists in SFT response")
        for row in lists:
            if not isinstance(row, list) or len(row) != horizon:
                raise ValueError("SFT response length does not match the configured horizon")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
                raise ValueError("SFT response values must be finite numbers")


def encode_example(tokenizer, messages, max_length):
    # Identical user chat prefix/options to Financial/SolarLLMProcess_utils.Qwen.
    text = tokenizer.apply_chat_template([messages[0]], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    response = tokenizer.encode(messages[1]["content"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide an EOS token")
    response.append(tokenizer.eos_token_id)
    if len(prefix) + len(response) > max_length:
        raise ValueError("Example exceeds qlora.max_length; increase it instead of truncating forecast labels")
    return {"input_ids": prefix + response, "attention_mask": [1] * (len(prefix) + len(response)),
            "labels": [-100] * len(prefix) + response}


def pad_examples(examples, pad_token_id):
    length = max(len(x["input_ids"]) for x in examples)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        padding = length - len(example["input_ids"])
        result["input_ids"].append(example["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(example["attention_mask"] + [0] * padding)
        result["labels"].append(example["labels"] + [-100] * padding)
    return result


class MessageDataset:
    def __init__(self, records, tokenizer, max_length):
        self.records, self.tokenizer, self.max_length = records, tokenizer, max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return encode_example(self.tokenizer, self.records[index]["messages"], self.max_length)


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_examples(examples, self.pad_token_id).items()}


def train(profile, records):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    config = profile["qlora"]
    base = profile["llm"]["base_model"]
    output = profile["llm"]["adapter_path"]
    tokenizer = AutoTokenizer.from_pretrained(base, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
    options = {"torch_dtype": dtype}
    if config["load_in_4bit"]:
        if not torch.cuda.is_available():
            raise ValueError("4-bit QLoRA needs CUDA; set qlora.load_in_4bit=false to use nonquantized LoRA")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(rank)
        options["device_map"] = {"": rank}
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(base, **options)
    if config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", target_modules="all-linear", r=config["rank"],
        lora_alpha=config["alpha"], lora_dropout=config["dropout"], bias="none"))
    arguments = TrainingArguments(
        output_dir=str(Path(output).parent / (Path(output).name + "_training")),
        num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        save_strategy="epoch", save_total_limit=2, logging_steps=10, report_to="none",
        remove_unused_columns=False, label_names=["labels"], seed=profile["common"]["seed"],
        ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=arguments,
                       train_dataset=MessageDataset(records, tokenizer, config["max_length"]),
                       data_collator=CompletionCollator(tokenizer), processing_class=tokenizer)
    trainer.train()
    trainer.save_model(output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        (Path(output) / "release_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--check-data", action="store_true", help="Validate the exported messages without loading a model")
    args = parser.parse_args(argv)
    profile = load_profile(args.config, args.horizon)
    if profile.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(profile["cuda_visible_devices"])
    # Match the original entry points' relative-path convention.
    os.chdir(ROOT)
    records = json.loads(Path(profile["qlora"]["train_file"]).read_text(encoding="utf-8"))
    validate_records(records, profile["domain"], profile["common"]["pred_len"])
    if args.check_data:
        print(json.dumps({"valid": True, "examples": len(records), "domain": profile["domain"]}))
    else:
        print(train(profile, records))


if __name__ == "__main__":
    main()
