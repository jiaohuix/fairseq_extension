import torch
import json
import torch.nn as nn
from fairseq.checkpoint_utils import torch_persistent_save
def load_state(filename):
    state = torch.load(filename)
    return state

def vocab_rearrange(ckpt,index=[]):
    index = torch.tensor(index)
    keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "decoder.output_projection.weight"]
    for key in keys:
        ckpt[key] = torch.index_select(ckpt[key],dim=0,index=index)
    return ckpt

class JsonTool:
    def load(self,filename):
        with open(filename,"r",encoding="utf-8") as f:
            data = json.load(f)
        return data

    def save(self,js_data,filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(js_data,f)
        print(f"save to {filename} success.")

if __name__ == '__main__':
    # params
    file = "F:/IKCEST/nllb_ckpt/checkpoint_last.pt"
    outfile = "checkpoint_last.pt"
    idx_file = "dict_joint_idx.json"
    # load index
    jstool = JsonTool()
    index = jstool.load(idx_file)["index"]
    # rearange vocab with index
    state = load_state(filename=file)
    state["model"] = vocab_rearrange(state["model"],index=index)
    # save state
    # torch_persistent_save(state,filename=outfile)
    vsize  = len(state["model"]["encoder.embed_tokens.weight"])
    print(vsize)