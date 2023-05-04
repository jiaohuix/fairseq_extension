from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel,
)
from transformers import AutoTokenizer, AutoModel
# params
model_name_or_path = "roberta-base"
peft_type = PeftType.LORA
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
# peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
# peft_config = LoraConfig(task_type="TOKEN_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)

# 加载lora模型
model = AutoModel.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 保存模型
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)         # 只保存lora，peft_model_id/adapter_config.json adapter_model.bin
model.base_model.save_pretrained(model_name_or_path) # 保存整个模型

# 加载lora模型
peft_config = PeftConfig.from_pretrained(peft_model_id)
print(peft_config)
model_name_or_path = "roberta-base"
modelx = AutoModel.from_pretrained(model_name_or_path) # 加载基础模型modelx
modelxlora = PeftModel.from_pretrained(modelx, peft_model_id) # 加载lora权重到modelx


# 自行保存配置和参数
peft_cfg_js = peft_config.__dict__
lora_dict = get_peft_model_state_dict(model)
lora_cfg = LoraConfig(**peft_cfg_js) # 加载配置
print(lora_cfg)

# adapters_weights = torch.load(
#     filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )
# load the weights into the model
# load the weights into the model
set_peft_model_state_dict(modelxlora, lora_dict)
# 还要设置inference...