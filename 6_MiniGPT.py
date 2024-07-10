# Scripting google colab work on; GPT2.0.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
# import requests

# hyperparameters
batch_size = 32 # how many independent sequences will be processed in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ability to use GPU
eval_iters = 200
# ---------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/brahma2024/LLM-study/main/shakespear.txt
with open('shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(f'{chars=}')
# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, returns a list of integers
decode = lambda l: ''.join(itos[i] for i in l) # decoder: takes a list of integers, returns a string

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest will be val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    # doing batch stacking, for each new batch starts at index from ix
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix]) # shifted by 1
    x, y = x.to(device), y.to(device) # When device = CUDA | when we load the data we move it to device
    return x, y

@torch.no_grad() # context manager torch.no_grad when used tells pytroch to not call .backward() function here
# this makes sure that pytorch will not build/save the computation graph for this function and will not save all the intermediate variables
# makes it much more memory efficient 
def estimate_loss(): # estimate_loss func averages the loss over multiple multiple batches | removes the noise | refer colab
    out = {}
    model.eval() # setting the model to eval phase | this is particularly important when we have batch_norm layer
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # reverting the model back to training phase
    return out


# Define the Model: Initializing the simplest language model - BigramLanguageModel
class BigramLanguagModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directy reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, input_batch, targets=None):

        # input_batch and taegts are both (B, T) tensor of integers
        logits = self.token_embedding_table(input_batch)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape # ( batch_size, block_size, embd_dim)
            logits = logits.view(B*T, C) # convert logits into 2D array: flatted out batch x time dimension into 1D | check F.cross_entropy docs
            targets = targets.view(B*T) # flatten out all yb into 1D array
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def train(self):
        pass

    def eval(self):
        pass

    def generate(self, x_in, max_new_token):
        # x_in is (B, T) aray of indices in the current context
        for _ in range(max_new_token):
            # get the predictions
            logits, loss = self(x_in) # get log probablities
            # focus only on the last timestep
            logits = logits[:, -1, :] # get probablities for the last character in context | select the last time_dim
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # get probablity for each possible character to come next
            # sample from the distribution
            idx = torch.multinomial(probs, num_samples=1, replacement=True) # predict next character index
            # append sampled index to the running sequence
            x_in = torch.cat((x_in, idx), dim=1) # add the new character at the end

        return x_in

model = BigramLanguagModel(vocab_size)
m = model.to(device) # when we create the model we move the model parameters to the device
# e.g. is the nn.Embedding table has a .weight which stores the lookup table | that is moved to the device i.e. GPU
# so all the calculations happen on the GPU | makes it a lot faster

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Start monitoring
# requests.get('http://localhost:5000/start_monitoring')

# train the model
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad() # set gradients to zero
    loss.backward() # compute the gradients in the backward pass
    optimizer.step() # udpate the model.parameters = lr x -grads

# Stop monitoring
# data = requests.get('http://localhost:5000/data').json()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
out = m.generate(context, max_new_token=500) # generating from the model on the device | this is the inference part
print(f'{len(out)=}')
for i in range(len(out)):
    print(decode(out[i].tolist()))