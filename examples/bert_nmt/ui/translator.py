'''

src lang, tgt lang

input area, output area
submit

params
https://pypi.org/project/translate/
参考：https://huggingface.co/spaces/Geonmo/nllb-translation-demo/blob/main/app.py
'''
from translate import Translator
import gradio as gr

language_codes = {
    "中文":"zh",
    "英文": "en",
    "日文": "ja"
}

def translate(text,src_lang,tgt_lang ,seed=1):
    src_lang = language_codes[src_lang]
    tgt_lang = language_codes[tgt_lang]
    translator = Translator(from_lang=src_lang,to_lang=tgt_lang) # 模型应该在外部加载，然后预热
    translation = translator.translate(text)
    return translation



def main():

    with gr.Blocks("Jiaohui Translator") as demo:
        # 描述信息，用markdown
        gr.Markdown(
            """
            角灰翻译器
            """)
        # 选择语言
        lang_codes = list(language_codes.keys())
        with gr.Row():
            # radio是单选框，使用下拉框请用dropdown
            # src_lang = gr.Radio(choices=LANGUAGE_TAG, value='zh', label='源语言')
            # tgt_lang = gr.Radio(choices=LANGUAGE_TAG, value='en', label='目标语言')
            src_lang = gr.Dropdown(choices=lang_codes,value="中文", label='源语言') # label是显示的标签
            tgt_lang = gr.Dropdown(choices=lang_codes, value="英文", label='目标语言')

        # 输入输出
        with gr.Row():
            with gr.Column():
                src_text = gr.Textbox(lines=14, placeholder="Source input", label="Input") # label是显示的标签
                with gr.Row(): # generate clear
                    gen = gr.Button("Generate")
                    clear = gr.Button("Clear")

            outputs = gr.Textbox(lines=15, label="Output")

        gr.Markdown(
            """
            生成参数
            """)

        with gr.Row():
            seed = gr.Slider(maximum=10000, value=8888, step=1, label='Seed')
            # 第二行 第三行
            with gr.Row():
                out_seq_length = gr.Slider(maximum=8192, value=128, minimum=1, step=1,
                                           label='Output Sequence Length')
                temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
            with gr.Row():
                top_k = gr.Slider(maximum=100, value=0, minimum=0, step=1, label='Top K')
                top_p = gr.Slider(minimum=0,maximum=1, value=0.95,label="TOP P ")

        # 输入输出处理逻辑
        inputs = [src_text,src_lang,tgt_lang,seed] # 只有用得到的参数，才能在界面条件数值大小
        gen.click(fn=translate, inputs=inputs, outputs=outputs)
        # clear.click(fn=lambda value: gr.update(value=""), inputs=clear, outputs=src_text) # TODO:清除只清除了输入，输出没有
        clear.click(lambda: ("", ""), outputs=[src_text, outputs], show_progress=True)
        # examples
        examples = [["很好，很有精神！","中文","英文"]]

        gr_examples = gr.Examples(examples=examples,inputs=[src_text,src_lang,tgt_lang])

    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)

if __name__ == '__main__':
    main()