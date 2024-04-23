from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline



infile = "datasets/test2.jsonl"
train_datasets = load_dataset('json', data_files=[infile],split="train")
print(train_datasets)
print(train_datasets[0])
src_lang = "zh"
tgt_lang = "en"

def filter_lang_pair(example):
    ''' 按照顺序过滤，避免两倍数据 '''

    lang =  f"{src_lang}-{tgt_lang}"
    # langs = [lang for lang, text in example["translation"].items() if text is not None]
    return lang == example["lang"]


# zh_en_translations = train_datasets.filter(lambda example: src_lang in example.keys() and tgt_lang in example.keys()) 
# zh_en_translations = train_datasets.filter(lambda example:   example[src_lang] is not None  and   example[tgt_lang] is not None) 
zh_en_translations = train_datasets.filter(lambda example: filter_lang_pair(example))# todo: zh-en en-zh会被变成2份
print(len(zh_en_translations["translation"]))
print(zh_en_translations)
print(zh_en_translations["translation"][-1])

# print(zh_en_translations["translation"][-100:])
# print(zh_en_translations)
# print(len(zh_en_translations))
# print(zh_en_translations["train"])
# print(zh_en_translations[0])

# .map(lambda example: example["translation"])  

# infile = "datasets/test.jsonl"
# train_datasets = load_dataset('json', data_files=[infile],split="train")
# # print(train_datasets)
# # print(train_datasets[0])
# src_lang = "en"
# tgt_lang = "zh"

# def valid_filter(example):
#     ''' 按照顺序过滤，避免两倍数据 '''
#     langs = [lang for lang, text in example.items() if text is not None]
#     print(langs)
#     return src_lang == langs[0]  and tgt_lang == langs[1] 


# # zh_en_translations = train_datasets.filter(lambda example: src_lang in example.keys() and tgt_lang in example.keys()) 
# # zh_en_translations = train_datasets.filter(lambda example:   example[src_lang] is not None  and   example[tgt_lang] is not None) 
# zh_en_translations = train_datasets.filter(lambda example: valid_filter(example))  # todo: zh-en en-zh会被变成2份

# print(len(zh_en_translations))
# print(zh_en_translations[0])


tokenizer = AutoTokenizer.from_pretrained("/mnt/f/down/m2m100_418M/")

# ### process the datasets for all language pairs
# lang_list = ['zh-fr','zh-ru','zh-th','zh-ar','zh-en',
#                 'fr-zh','ru-zh','th-zh','ar-zh','en-zh'
#             ]
lang_list= ['zh-en','en-zh']
# train_datasets_list = []


# print(len(train_datasets['train'])) 



for ll in lang_list:
    source_lang = ll.split('-')[0]
    target_lang = ll.split('-')[1]
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=1024, padding=True, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=1024, padding=True, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.

        model_inputs["labels"] = labels["input_ids"]
        forced_bos_list = [forced_bos_token_id for i in range(len(model_inputs['labels']))]
        model_inputs["forced_bos_token_id"] = forced_bos_list
        return model_inputs

    train_datasets = load_dataset('json', data_files=[infile])
    # train_datasets = train_datasets.filter(lambda example: example[source_lang] is not None  and   example[target_lang] is not None) 
    train_datasets = train_datasets.filter(lambda example: filter_lang_pair(example))  # todo: zh-en en-zh会被变成2份

    print(len(train_datasets['train'])) 
    column_names = train_datasets["train"].column_names
    # column_names = [source_lang, target_lang]
    print(column_names)
    column_names = ["translation","lang"]
    print("process")
    print("column_names",column_names)
    train_datasets_token = train_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,    
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    print(train_datasets)
    print(train_datasets_token)