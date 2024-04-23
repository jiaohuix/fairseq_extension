遇到的问题：
1 m2mtranier不存在,用Seq2SeqTrainer替代
2 metric不行
ConnectionError: Couldn't reach https://raw.githubusercontent.com/huggingface/datasets/2.13.0/metrics/sacrebleu/sacrebleu.py (ConnectionError(MaxRetryError("HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Max retries exceeded with url: /huggingf
源码安装evaluate，然后evaluate.load而不是load_metric
3 accelerate==0.27.2
https://github.com/huggingface/transformers/issues/29216
4 apex源码安装


5 训练速度：
测试集
2000*2 *3 =12000
bsz=8 5.4it/s
data_size * epoch / bsz(非梯度累加) / speed
train:
450000*2 * 3  / 8 / 5.4 


bsz变大：
bsz=16 4.8it/s
data_size * epoch / bsz(非梯度累加) / speed
train:
450000*2 * 3  / 16 / 4.8

200000*3/10/64

/mnt/f/workspace/nmt/ckpt/m2m_ft_ikcest/ 

6 日志
https://docs.wandb.ai/guides/integrations/huggingface#3-log-your-training-runs-to-wb

7 apex
https://stackoverflow.com/questions/66610378/unencryptedcookiesessionfactoryconfig-error-when-importing-apex

8 apex没有，然后报fp16问题，
从源码下载transformers peft

9 load_dataset失败


        import pandas as pd
        from datasets import Dataset  
        df = pd.read_json(infile, lines=True)
        train_datasets = Dataset.from_pandas(pd.read_json(data_args.train_file, lines=True))  
        train_datasets = Dataset.from_pandas(pd.read_json(data_args.validation_file, lines=True))  

10 deeps cuda home 
https://github.com/microsoft/DeepSpeed/issues/2772
➜ export CUDA_HOME=/usr/local/cuda-11.2 #your cuda installed path
➜ pip install deepspeed


nvcc --version
>> Command 'nvcc' not found, but can be installed with:
apt install nvidia-cuda-toolkit
which nvcc
>> /usr/bin/nvcc
export CUDA_HOME=/usr/
pip install  deepspeed==0.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

11 /save_and_load.py:141: UserWarning: Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`
https://github.com/huggingface/peft/issues/349
不用手动修改，已经自动改过了
https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py#L142



12 dora:
use_dora=True
https://github.com/huggingface/peft/releases/tag/v0.9.0
We now support Weight-Decomposed Low-Rank Adaptation aka DoRA via #1474. This new method is builds on top of LoRA and has shown very promising results. Especially at lower ranks (e.g. r=8), it should perform much better than LoRA. Right now, only non-quantized nn.Linear layers are supported. If you'd like to give it a try, just pass use_dora=True to your LoraConfig and you're good to go.

https://huggingface.co/papers/2402.09353


13 peft在全量微调时，加载报错：
RuntimeError: Error(s) in loading state_dict for PeftModel:
        size mismatch for base_model.model.lm_head.modules_to_save.default.weight: copying a param with shape torch.Size([128104, 1024]) from checkpoint, the shape in current model is torch.Size([128112, 1024]).
/home/jiahui/workspace/nmt/thesis_nmt/scripts/eval.py:40: FutureWarning: Passing literal json to 'read_json' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object.

128104少了8个.

        tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        print("len tokenizer:", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer)) # 防止miss match，tokenizer应该用新的
        