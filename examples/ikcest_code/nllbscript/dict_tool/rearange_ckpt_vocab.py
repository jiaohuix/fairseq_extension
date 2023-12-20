'''
功能：输入权重和索引，用索引重排 emed和out_proj,并保存新权重
'''
import torch
import json
import sys
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
    assert len(sys.argv)==4,f"usage: python {sys.argv[0]} <state_path> <save_path>  <index_path> "
    # params
    state_path = sys.argv[1]
    save_path =  sys.argv[2]
    index_path =  sys.argv[3]
    # load index
    jstool = JsonTool()
    index = jstool.load(index_path)["index"]
    # rearange vocab with index
    state = load_state(filename=state_path)
    state["model"] = vocab_rearrange(state["model"],index=index)
    # save state
    torch_persistent_save(state,filename=save_path)
