# Scripting google colab work on; GPT2.0.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will be processed in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaal_iters = 200
# ---------------

torch.manual(1337)

# wget https://raw.githubusercontent.com/brahma2024/LLM-study/main/shakespear.txt
with open('shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {i:s for i, s in enumerate(chars)}
itos = {s:i for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, returns a list of integers
decode = lambda l: ''.join(itos[i] for i in l) # decoder: takes a list of integers, returns a string

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest will be val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randn(len(data_source) - block_size, (batch_size,))

    # doing batch stacking, for each new batch starts at index from ix
    x = torch.stack(data_source[i:i+block_size] for i in ix)
    y = torch.stack(data_source[i+1:i+block_size+1] for i in ix) # shifted by 1

    return x, y


# Define the Model: Initializing the simplest language model - BigramLanguageModel
class BigramLanguagModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

