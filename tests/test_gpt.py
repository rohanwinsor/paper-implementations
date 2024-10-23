import os

# os.chdir('../')
print(os.getcwd())
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tiktoken
from gpt import GPTModel, GPTConfig

from utils.utils import generate, load_weights_into_gpt
from utils.gpt_download import download_and_load_gpt2

tokenizer = tiktoken.get_encoding("gpt2")
txt1 = "Every effort moves you"
gpt_config = GPTConfig()
# ## Load Open AI Weights
settings, params = download_and_load_gpt2(model_size="124M", models_dir="models/gpt2")
model = GPTModel(gpt_config)
load_weights_into_gpt(model, params)
model.eval()
batch = torch.stack(
    [torch.tensor(tokenizer.encode(txt1))],
    dim=0,
)

logits = generate(batch, model, gpt_config.context_length, 25, 1.5, 50)
print(tokenizer.decode(logits.tolist()[0]))
