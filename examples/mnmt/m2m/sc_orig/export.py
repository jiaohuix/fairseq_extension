import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig,AutoPeftModelForSeq2SeqLM, LoraModel, get_peft_model


lora_id = "/mnt/f/workspace/nmt/ckpt/ikcest_mft_lora/checkpoint-20000/"
output_dir = "/mnt/f/workspace/nmt/ckpt/ikcest_mft_lora/checkpoint-20000-full"
peft_config = PeftConfig.from_pretrained(lora_id) 
base_model_path = peft_config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)

peft_config = PeftConfig.from_pretrained(lora_id)
model = get_peft_model(model, peft_config)
model.merge_and_unload()

model.save_pretrained(output_dir)
torch.save(model.state_dict(),os.path.join(output_dir,"pytorch_model.bin"))