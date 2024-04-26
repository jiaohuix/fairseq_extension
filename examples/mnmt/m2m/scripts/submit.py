import json
import os
import sys
import shutil
import argparse
# 打包rst文件为zip文件
import zipfile

# 定义需要处理的源语言和目标语言对
lang_pairs = {
    ('fr', 'zh'),
    ('ru', 'zh'),
    ('th', 'zh'),
    ('zh', 'fr'),
    ('zh', 'ru'),
    ('zh', 'th')
}


# 创建或清空.rst文件
def create_or_clear_rst_file(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('')

    # 读取jsonl文件并保存翻译结果到.rst文件


def save_translations_to_rst(jsonl_path, rst_folder):
    translations = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            src_lang = data['src_lang']
            tgt_lang = data['tgt_lang']
            pred = data['pred']
            if (src_lang, tgt_lang) in lang_pairs:
                filename = f"{src_lang}_{tgt_lang}.rst"
                file_path = f"{rst_folder}/{filename}"
                # create_or_clear_rst_file(file_path)
                with open(file_path, 'a', encoding='utf-8') as rst_file:
                    rst_file.write(pred + '\n')

                # 入口函数


def main(jsonl_path, rst_folder):
    # 确保rst文件夹存在
    if not os.path.exists(rst_folder):
        os.makedirs(rst_folder)
        # 读取jsonl文件并保存翻译结果
    save_translations_to_rst(jsonl_path, rst_folder)





def zip_rst_files(rst_folder, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(rst_folder):
            for file in files:
                if file.endswith('.rst'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, rst_folder))

                # 压缩rst文件为zip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate texts using m2m100 model and save results to a JSON file")
    parser.add_argument("-i","--infile", type=str, default=None, help="pred file path(predict.py output jsonl fine)")
    parser.add_argument("-o","--outdir", type=str, default="results", help="Output dir")
    args = parser.parse_args()
    

    # 假设jsonl文件名为translations.jsonl，rst文件保存在results文件夹中
    jsonl_path = args.infile
    rst_folder = args.outdir
    # 创建输出目录
    if not os.path.exists(rst_folder):
        os.makedirs(rst_folder)
    else:
        # 如果目录存在，清空目录内容
        shutil.rmtree(rst_folder)
        os.makedirs(rst_folder)
    main(jsonl_path, rst_folder)
    zip_filename = os.path.join(rst_folder,os.path.basename(args.outdir) + ".zip")
    zip_rst_files(rst_folder, zip_filename)
    