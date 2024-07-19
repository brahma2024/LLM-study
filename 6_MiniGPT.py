# Scripting google colab work on; GPT2.0.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
# import requests

# hyperparameters
batch_size = 64 # how many independent sequences will be processed in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ability to use GPU
eval_iters = 200
n_embd = 384
dropout = 0.2 # every fwd and backward pass 20% of the intermediate calculations are disabled and dropped
n_head = 6 # n_embd // n_head = head_size | 384 // 6 = 64
n_layer = 6
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

# Define the Head module
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Note: tril is not a parameter of the module, so as per pytorch naming convention its called a buffer
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # lower triangular matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # forward method builds the computation graph
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention score (affinites)
        wei = q @ k.transpose(-2, -1) # transpose the time x channel dimension -> channel x time | inside a batch (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei * C**-0.5 # scale down by sqrt(fan_in) | fan_in = head_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, C)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C) # these are the probabilites for prediction
        return out

# creating the module for Multi-head Attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # forward method builds the computation graph
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating multiple heads of self-attention on the channel dimension
        out = self.dropout(self.proj(out)) # projection is just the linear transformation of the outcome of the out layer

        return out # this is the projection back into the residual pathway

# Position-wise Feedforward Networks
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity: adding computation in the feed-forward """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,  n_embd * 4), # in the paper n_embd = 512 | and inner-layer has dimensionality = 2048 = n_embd * 4
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd), # here the second Linear layer acts as a linear transformation layer going back into the linear pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # forward method builds the computation graph
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # B, T acts as batch dimensions | LayerNorm just normalizes the feature-vetor or channel/embedding vetor
        self.ln2 = nn.LayerNorm(n_embd) # LayerNorm makes feature vector unit-norm, unit-var

    def forward(self, x):
        # forward method builds the computation graph
        # the x + implementation is implementing residual connections, think of it like this:

        # we fork-off do some communication and then plug back into main branch
        # PreNorm | LayerNorm: is applied on x before it goes into multi-head attention | feed-forward
        x = x + self.sa(self.ln1(x)) # apply one head self-attention. (B, T, C) | this is the communication layer, where inforamtion gets communicated
        # we fork-off do some computation and then plug back into main branch
        x = x + self.ffwd(self.ln2(x)) # (B, T, C) # this is the computation layer, when the information gathered from self-attention is computed upon | see RELU
        return x


# Define the Model: Initializing the simplest language model - BigramLanguageModel
class BigramLanguagModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directy reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim= n_embd) # token embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # positional encoding table

        # self.blocks: this becomes a deep NN, and to mitigate that we will use Residual connections + LayerNorm
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self. ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head = language model head | takes embedding table as input and returns vocab_size logits 

    def forward(self, idx, targets=None):
        # forward method builds the computation graph
        B, T = idx.shape

        # x-in and targets are both (B, T) tensor of integers
        token_embedding = self.token_embedding_table(idx) # (B, T, C) | C = n_embd
        position_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) | Create position embeddings for each position
        x = token_embedding + position_embedding # pos_embd gets right alighned and broadcasted across batches in tok_emb
        # now x holds not just the token identities but also the position at which these tokens occur
        # this positional information will not matter unless self-attention is implemented, because the positional information does not matter still
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, C) | C = vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape # ( batch_size, block_size, embd_dim)
            logits = logits.view(B*T, C) # convert logits into 2D array: flatted out batch x time dimension into 1D | check F.cross_entropy docs
            targets = targets.view(B*T) # flatten out all yb into 1D array
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    

    def generate(self, idx, max_new_token):
        # idx is (B, T) aray of indices in the current context
        for _ in range(max_new_token):
            # crop idx to the last block_size tokens
            idx_cropped = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cropped) # get log probablities
            # focus only on the last timestep
            logits = logits[:, -1, :] # get probablities for the last character in context | select the last time_dim
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # get probablity for each possible character to come next
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True) # predict next character index
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # add the new character at the end

        return idx

model = BigramLanguagModel()
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
out = m.generate(context, max_new_token=1000) # generating from the model on the device | this is the inference part
print(decode(out[0].tolist()))