import torch
import torch.nn as nn
import loralib as lora

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class Mlp(nn.Module):
  def __init__(self,in_features, out_features):
    super(Mlp,self).__init__()
    self.layer1 = nn.Linear(in_features, in_features*4)
    self.layer2 = nn.Linear(in_features*4, out_features)

  def forward(self,x):
    return self.layer2(self.layer1(x))

# lora
class Mlp2(nn.Module):
  def __init__(self,in_features, out_features):
    super(Mlp2,self).__init__()
    self.layer1 = nn.Linear(in_features, in_features*4)
    self.lora_layer2 = lora.Linear(in_features*4, out_features) # 定义都不一样了，权重不能加载

  def forward(self,x):
    return self.lora_layer2(self.layer1(x))


def load_lora_model(ptm_state_dict, lora_model, lora_state_dict = None):
    # key -> lora_key
    lora_state_rand = lora_model.state_dict()
    for key,weight in lora_state_rand.items():
        ptm_key = key.replace("lora_","")
        if key in ptm_state_dict.keys():
            lora_state_rand[key] = ptm_state_dict[key]
        elif ptm_key in ptm_state_dict.keys():
            lora_state_rand[key] = ptm_state_dict[ptm_key]
        else:
            print(f"key {key} load not successful.")
    print("loading ptm ckpt...")
    lora_model.load_state_dict(lora_state_rand , strict=False)
    if lora_state_dict is not None:
        print("loading lora ckpt...")
        lora_model.load_state_dict(lora_state_dict, strict=False)
    print("over...")

if __name__ == '__main__':

    bsz=4
    indim=10
    outdim=20
    x = torch.randn(bsz,indim)
    model_raw = Mlp(indim,outdim)
    # res = model_raw(x)
    # print(res.shape)
    print_trainable_parameters(model_raw)
    checkpoint_path="mlp_raw.pt"
    torch.save(model_raw.state_dict(), checkpoint_path)

    model_lora = Mlp2(indim,outdim)
    res = model_lora(x)
    # print(res.shape)
    print_trainable_parameters(model_lora) # 100%
    lora.mark_only_lora_as_trainable(model_lora) # 与nn.Linear和lora.Linear无关，只会看参数名含lora_
    print_trainable_parameters(model_lora) # 65%
    print(model_lora)
    # save
    checkpoint_path = "mlp_lora.pt"
    torch.save(lora.lora_state_dict(model_lora), checkpoint_path) # 只保存lora的参数

    # 加载参数
    # Load the pretrained checkpoint first
    model_lora.load_state_dict(torch.load('mlp_raw.pt'), strict=False,) # layer2没加载
    # Then load the LoRA checkpoint
    model_lora.load_state_dict(torch.load('mlp_lora.pt'), strict=False) # 加载了自己的layer2

    # 现在我熟悉了loralib的使用，一个巨大的问题是loralib，需要修改模型定义，甚至参数名，layer2->lora_layer2，这样在第一次加载预训练模型参数时，lora_x的没有加载，
    # 后续训练了lora部分后，再保存加载lora部分就没问题了，关键是第一次。。。。

    # save只要保存lora；  那第一次加载呢？
    # 已经解决参数名不一致导致初始加载预训练权重错误问题； 可不可以在外部修改lora模型呢？必须要修改注意力？

    # TODO: 不修改参数名，而是修改nn为lora，加载或者冻结的时候，判断类型？参考peft，需要loraconfig来指定需要修改的模块

    # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py