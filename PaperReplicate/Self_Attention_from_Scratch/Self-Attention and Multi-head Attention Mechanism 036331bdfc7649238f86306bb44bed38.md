# Self-Attention and Multi-head Attention Mechanism

Tags: Recap

# Why Self-Attention?

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled.png)

- The concept of “attention” in deep learning [has its roots in the effort to improve Recurrent Neural Networks (RNNs)](https://arxiv.org/abs/1409.0473) for handling longer sequences or sentences.
    - Translating a sentence word-by-word does not work effectively.
- To overcome this issue, attention mechanisms were introduced to give access to all sequence elements at each time step. **The key is to be selective and determine which words are most important in a specific context**.
    - In 2017, the transformer architecture introduced a standalone self-attention mechanism, eliminating the need for RNNs altogether.

# What is Self-Attention?

- We can think of self-attention as a mechanism that **enhances the information content of an input embedding by including information about the input’s context**. In other words, the self-attention mechanism **enables the model to weigh the importance of different elements in an input sequence and dynamically adjust their influence on the output**.
    - This is especially important for language processing tasks, where the meaning of a word can change based on its context within a sentence or document.

# How to Define Self-Attention?

## Embedding Input Sentence

- For simplicity, here our dictionary dc is restricted to the words that occur in the input sentence. In a real-world application, we would consider all words in the training dataset (typical vocabulary sizes range between 30k to 50k).

```python
sentence  = "Life is short, eat dessert first"

# Create Dictionary
dict = {s : i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}
# dict: {'Life': 0, 'dessert': 1, 'eat': 2, 'first': 3, 'is': 4, 'short': 5}

import torch
sentence_idx = torch.tensor([dict[s] for s in sentence.replace(',', '').split()])
# sentence_idx: tensor([0, 4, 5, 2, 1, 3])
```

### Word Embedding

- Here, we will use a 16-dimensional embedding such that each input word is represented by a 16-dimensional vector.

```python
torch.manual_seed(123)
embeder = torch.nn.Embedding(6, 16)
embedded_sentence = embeder(sentence_idx).detach() # [6, 16]
```

## Define Unnormalized Attention Weights

### Define Weight Matrices

- Self-Attention uses three weight matrices, referred to as $W_q, W_k, W_v$, which are adjusted as model parameters during training.
    - These matrics serve to project the inputs into query , key, and value components of the sequence.

$$\text{Query Sequence: } \mathbf{q}^{(\mathrm{i})}=\mathbf{W}_{\mathrm{q}} \mathbf{x}^{(\mathrm{i})} \text { for } \mathrm{i} \in[1, \mathrm{~T}]$$


$$\text{Key Sequence: } \mathbf{k}^{(\mathrm{i})}=\mathbf{W}_{\mathrm{k}} \mathbf{x}^{(\mathrm{i})} \text { for } \mathrm{i} \in[1, \mathrm{~T}]$$

$$\text{Value Sequence: } \mathbf{v}^{(\mathrm{i})}=\mathbf{W}_{\mathrm{v}} \mathbf{x}^{(\mathrm{i})} \text { for } \mathrm{i} \in[1, \mathrm{~T}]$$

$$\text{i refers to the token index position in the input sequence}$$


- Here, both $q^{(i)}, k^{(i)}$ are vectors of dim $d_k$; $v^{(i)}$ is the vector of dim $d_v$.
$W_q,\ W_k$ have shape $d_k \times d$, $W_v$ has shape $d_v \times d$, $d$ is the embedding dim of each word vector $x^{(i)}$
- Since we are computing the dot-product between the query and key vectors, these two vectors have to contain the same number of elements ($d_q=d_k$). However, the number of elements in the value vector $d_v$ which determines the size of the resulting context vector, is arbitrary.

```python
torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.nn.Parameter(torch.randn(d_q, d)) # [24, 16]
W_key = torch.nn.Parameter(torch.randn(d_k, d)) # [24, 16]
W_value = torch.nn.Parameter(torch.randn(d_v, d)) # [28, 16]

```

### Compute the unnormalized weights

We pick the second words $x^{(2)}$ as example

```python
x_2 = embedded_sentence[1]
query_2 = W_query @ x_2 # [24]
key_2 = W_key @ x_2 # [24]
value_2 = W_value @ x_2 # [28]

# compute the remaining key-value for all inputs
keys = (W_key @ embedded_sentence.T).T # [6, 24]
values = (W_value @ embedded_sentence.T).T #[6, 28]

```

Then compute $\omega_{\mathrm{ij}}=\mathbf{q}^{(\mathrm{i})^{\top}} \mathbf{k}^{(\mathrm{j})}$

```python
# compute unnormalized attention weight for the query and 5th input element
omega_24 = query_2.dot(keys[4]) # tensor(-98.1709, grad_fn=<DotBackward0>)
# then we can let the 2nd word asks every other words
omega_2 = query_2 @ keys.T # (tensor([  83.1533,   95.5014, -100.8583,   63.5880,  -98.1709,    9.3997], grad_fn=<SqueezeBackward3>)

```

## Computing Attention Score

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%201.png)

- Then we need the normalized attention weights $\alpha$  by applying the softmax function.
    - $1/\sqrt d_k$ is used to scale $w$ before normalization, so that we can ensure that the Euclidean length of the weight vectors will be approximately in the same magnitude.
    - This helps prevent the attention weights from becoming too small or too large, which could lead to numerical instability or affect the model’s ability to converge during training.

```python
import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)  
# tensor([7.4329e-02, 9.2430e-01, 3.6185e-18, 1.3699e-03, 6.2628e-18, 2.1523e-08],grad_fn=<SoftmaxBackward0>)

```

## Compute Context Vector

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%202.png)

- $z^{(2)}$ is an attention-weighted version of our original query input $x^{(2)}$
- The context vector represents the second word in the context of the entire sentence and can be used as input to subsequent layers in a neural network or other models.

```python
context_vector_2 = attention_weights_2 @ values # [28]
```

## Dimension Graph

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%203.png)

## Functionalized Version

```python
def calc_attn(embedded_sentence: torch.Tensor,
              dim_q: int,
              dim_k: int,
              dim_v: int,
              ):
    assert dim_k == dim_q, "dim(K) == dim(Q) must be met!"

    hid_dim = embedded_sentence.shape[-1]
    W_Q = nn.Parameter(torch.randn(dim_q, hid_dim))
    W_K = nn.Parameter(torch.randn(dim_k, hid_dim))
    W_V = nn.Parameter(torch.randn(dim_v, hid_dim))

    Q = (W_K @ embedded_sentence.T).T  # (T, d_q)
    K = (W_Q @ embedded_sentence.T).T  # (T, d_k)
    V = (W_V @ embedded_sentence.T).T  # (T, d_v)

    attn = torch.softmax(1 / dim_k ** .5 * Q @ K.T, dim = 1) @ V # (T, d_v)

    return attn
```

# Multi-Head Attention

## Single-Head Attention

In the scaled dot-product attention, the input sequence was transformed using three matrices representing the query, key, and value. 

- These three matrices can be considered as a single attention head in the context of multi-head attention.

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%204.png)

## Multi-Head Attention

As its name implies, multi-head attention involves multiple such heads, each consisting of query, key, and value matrices.

- The final dimension of the context vector should be $(head, d_v)$

![Untitled](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%205.png)

### Functionalized Version

```python
def multi_head_attn(embedded_sentence: torch.Tensor,
                    head: int,
                    dim_q: int,
                    dim_k: int,
                    dim_v: int):
    assert dim_k == dim_q, "dim(K) == dim(Q) must be met!"
    hid_dim = embedded_sentence.shape[-1] # dim
    stacked_embedded_sentence = embedded_sentence.repeat(head, 1, 1) # (head, T, dim)

    W_Q = nn.Parameter(torch.randn(head, dim_q, hid_dim)) # (h, d_q, dim)
    W_K = nn.Parameter(torch.randn(head, dim_k, hid_dim)) # (h, d_k, dim)
    W_V = nn.Parameter(torch.randn(head, dim_v, hid_dim)) # (h, d_v, dim)

    Q = (W_Q @ stacked_embedded_sentence.transpose(1, 2)).transpose(1, 2) # (h, T, d_q)
    K = (W_K @ stacked_embedded_sentence.transpose(1, 2)).transpose(1, 2) # (h, T, d_k)
    V = (W_V @ stacked_embedded_sentence.transpose(1, 2)).transpose(1, 2) # (h, T, d_v)

    mha = torch.softmax(1 / d_k ** .5 * Q @ K.transpose(1, 2), dim = 2) @ V # (h, T, d_v)

    return mha
```

### For Single Token

```python
head = 3

# (3, 24, 16)
multihead_W_query = torch.nn.Parameter(torch.randn(head, d_q, d))
multihead_W_key = torch.nn.Parameter(torch.randn(head, d_k, d))
# (3, 28, 16)
multihead_W_value = torch.nn.Parameter(torch.randn(head, d_v, d))

# q, k, v for x_2
multihead_query_2 = multihead_W_query @ x_2 # (3, 24)
multihead_key_2 = multihead_W_key @ x_2 # (3, 24)
multihead_value_2 = multihead_W_value @ x_2 # (3, 24)

# x_2 asks each other words, then we need to calculate the k, v for other tokens
# first, we need to expand the input sequence embeddings to the number of heads
stacked_inputs = embedded_sentence.T.repeat(head, 1, 1) # (3, 16, 6)

multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)# (3, 24, 6)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs) # (3, 28, 6)

# let x_2 asks every token -> unnormalized attention score
multihead_query_2.unsqueeze(1)
multihead_attention_unnormalized_score_2 =  torch.bmm(multihead_query_2.unsqueeze(dim = 1),
                                                      multihead_keys).squeeze() # (3, 6)
# normalized multihead attention score
multihead_attention_normalized_score_2 = F.softmax(multihead_attention_unnormalized_score_2 / d_k
                                                   ** 0.5, dim = 1) # (3, 6)
# multihead attention context score 
multihead_context_score_2 = torch.bmm(multihead_attention_normalized_score_2.unsqueeze(1),
                                      multihead_values.permute(0, 2, 1)).squeeze()  # (3, 28)

```

# Cross Attention

- In self-attention, we work with the same input sequence. In cross-attention, we mix or combine two *different* input sequences.
- Two input sequence $x_1$ and $x_2$ can have different numbers of elelments. However, their **embedding dimensions must match**

![If we set $x_1$ = $x_2$, it is equal to self-attention](Self-Attention%20and%20Multi-head%20Attention%20Mechanism%20036331bdfc7649238f86306bb44bed38/Untitled%206.png)

If we set $x_1$ = $x_2$, it is equal to self-attention

- **Generated tokens** from Decoder (sen0) **query** the **tokens from Encoder** (sent1)
    - Queries usually come from the decoder, and keys and values usually come from the encoder.

## Functionalized Version

```python
def cross_attn(embedded_sentence_0: torch.Tensor,
               embedded_sentence_1: torch.Tensor,
               dim_q: int,
               dim_k: int,
               dim_v: int):
    # Generated tokens from the decoder sentence 0 query the tokens from the encoder sentence 1
    assert dim_k == dim_q, "dim(K) == dim(Q) must be met!"
    assert embedded_sentence_0.shape[-1] == embedded_sentence_1.shape[-1], "Hid_dim must be same between two sentences"

    hid_dim = embedded_sentence_0.shape[-1]

    W_Q = nn.Parameter(torch.randn(dim_q, hid_dim))
    W_K = nn.Parameter(torch.randn(dim_k, hid_dim))
    W_V = nn.Parameter(torch.randn(dim_v, hid_dim))

    Q = (W_K @ embedded_sentence_0.T).T # (T_0, d_q)
    K = (W_Q @ embedded_sentence_1.T).T # (T_1, d_k)
    V = (W_V @ embedded_sentence_1.T).T # (T_1, d_v)

    attn = torch.softmax(1 / dim_k ** .5 * Q @ K.T, dim = 1) @ V # (T_0, d_v)

    return attn
```
