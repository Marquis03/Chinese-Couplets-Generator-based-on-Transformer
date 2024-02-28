import random
import joblib

import torch
import gradio as gr

from dataset import Vocab
from model import CoupletsTransformer

data_path = "./data/fixed_couplets_in.txt"
vocab_path = "./trained/vocab.pkl"
model_path = "./trained/CoupletsTransformer_best.pth"


vocab = joblib.load(vocab_path)
vocab_size = len(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CoupletsTransformer(
    vocab_size,
    d_model=256,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=1024,
    dropout=0.1,
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

example = (
    line.replace(" ", "").strip() for line in iter(open(data_path, encoding="utf8"))
)
example = [line for line in example if len(line) > 5]

example = random.sample(example, 300)


def generate_couplet(vocab, model, src_text):
    if not src_text:
        return "上联不能为空"
    out_text = model.generate(src_text, vocab)
    return out_text


input_text = gr.Textbox(
    label="上联",
    placeholder="在这里输入上联",
    max_lines=1,
    lines=1,
    show_copy_button=True,
    autofocus=True,
)

output_text = gr.Textbox(
    label="下联",
    placeholder="在这里生成下联",
    max_lines=1,
    lines=1,
    show_copy_button=True,
)

demo = gr.Interface(
    fn=lambda x: generate_couplet(vocab, model, x),
    inputs=input_text,
    outputs=output_text,
    title="中文对联生成器",
    description="输入上联，生成下联",
    allow_flagging="never",
    submit_btn="生成下联",
    clear_btn="清空",
    examples=example,
    examples_per_page=50,
)

demo.launch()
