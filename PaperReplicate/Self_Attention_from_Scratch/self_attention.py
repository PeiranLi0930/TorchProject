import torch
from torch import nn

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


if __name__ == '__main__':
    sentence = "Life is short, eat dessert first"
    # Create Dictionary
    dict = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}
    # print(dict)

    sentence_index = torch.tensor([dict[s] for s in sentence.replace(",", "").split()])
    # print(sentence_index)

    # word embedding
    embeder = torch.nn.Embedding(6, 16)
    embedded_sentence = embeder(sentence_index).detach()
    # print(embedded_sentence.shape)
    d_q, d_k, d_v = 24, 24, 30
    multi_head_attn(embedded_sentence, 3, d_q, d_k, d_v)









