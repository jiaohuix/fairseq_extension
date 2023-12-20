

# exposure bias

methods:

- 1 schedule sample
- 2 target nosing
- 3 rl
- 4 minimum risk training



Scheduled Sampling Based on Decoding Steps for Neural Machine Translation (EMNLP-2021 main conference)

https://github.com/Adaxry/ss_on_decoding_steps.

微信20： https://aclanthology.org/2020.wmt-1.24.pdf

微信21: https://aclanthology.org/2021.wmt-1.23.pdf

微信22: https://underline.io/lecture/67180-summer-wechat-neural-machine-translation-systems-for-the-wmt22-biomedical-translation-task

```
Target Denoising (Meng et al., 2020). In the
training stage, the model never sees its own errors. Thus the model trained with teacher-forcing
is prune to accumulated errors in testing (Ranzato
et al., 2016). To mitigate this training-generation
discrepancy, we add noisy perturbations into decoder inputs when finetuning. Thus the model becomes more robust to prediction errors by target
denoising. Specifically, the finetuning data generator chooses 30% of sentence pairs to add noise,
and keeps the remaining 70% of sentence pairs unchanged. For a chosen pair, we keep the source
sentence unchanged, and replace the i-th token of
the target sentence with (1) a random token of the
current target sentence 15% of the time (2) the
unchanged i-th token 85% of the time.
```



```
import random

def target_denoising_finetune(sentence_pairs):
    noisy_pairs = []
    clean_pairs = []

    for source, target in sentence_pairs:
        if random.random() < 0.3:  # 30% probability to add noise
            noisy_target = list(target)
            for i in range(len(target)):
                if random.random() < 0.15:  # 15% probability to replace with random token
                    noisy_target[i] = random.choice(target)
            noisy_pairs.append((source, ''.join(noisy_target)))
        else:
            clean_pairs.append((source, target))

    return clean_pairs + noisy_pairs
```

uie论文也提到了es





领域适应：https://zhuanlan.zhihu.com/p/370390321

https://xueqiu.com/9217191040/130638237

vivo: https://www.modb.pro/db/126159

对抗？

