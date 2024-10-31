import torch
from torch import nn
import math
import numpy as np
from einops import rearrange

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
        
    @staticmethod
    def backward(ctx, grad_output):
        #straight through estimator is unstable
        #return grad_output
        input, = ctx.saved_tensors
        # try tanh derivative
        return (1 - torch.square(torch.tanh(input))) * grad_output

def ste_sign(input):
    return SignSTE.apply(input)

DEFAULT_USA_CFG = {
    'lth_int_dim' : 128,
    'lth_final_dim': 32,
    'lth_hard_inference': False,
}

class USA(nn.Module):
    def __init__(self, num_heads, head_dim, usa_params = DEFAULT_USA_CFG):
        super(USA, self).__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.int_dim = usa_params['lth_int_dim']
        self.lth_final_dim = usa_params['lth_final_dim']
        self.lth_hard_inference = usa_params['lth_hard_inference']
        self.lth_thold = usa_params['lth_thold']
        self.learning_to_hash_transformation_k = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, self.int_dim), 
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.lth_final_dim)
                                    ) for i in range(self.num_heads)])
        self.learning_to_hash_transformation_q = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.lth_final_dim)
                                      ) for i in range(self.num_heads)])
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


    def forward(self, K, Q, hard=False):

        b,a,sk,d = K.shape
        _,_,sq,d = Q.shape

        Klifted = torch.zeros((b,a,sk,self.lth_final_dim), device=K.device)
        Qlifted = torch.zeros((b,a,sq,self.lth_final_dim), device=Q.device)
        
        for i in range(self.num_heads):
            Klifted[:,i,:,:] = self.learning_to_hash_transformation_k[i](K[:,i,:,:])
            Qlifted[:,i,:,:] = self.learning_to_hash_transformation_q[i](Q[:,i,:,:])

        if hard:
            Q = ste_sign(Qlifted)
            K = ste_sign(Klifted)
        else:
            Q = torch.tanh(Qlifted)
            K = torch.tanh(Klifted)

        bsz, _, q_seq_len, _ = Q.size()
        _, _, k_seq_len, _ = K.size()

        q = rearrange(Q, 'b h t d -> (b h) t d')
        k = rearrange(K, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        span_scores = torch.empty(bsz * self.num_heads, q_seq_len, k_seq_len, dtype=Q.dtype,
                                   device=Q.device)

        span_scores = rearrange(torch.baddbmm(span_scores, q, k, beta=0, alpha=1.0),
                                 '(b h) t s -> b h t s', h=self.num_heads)

        # mask 
        query_length, key_length = Q.size(-2), K.size(-2)
        causal_mask = torch.tril(torch.ones(query_length,key_length,device=K.device), diagonal=key_length - query_length).bool()
        mask_value = torch.finfo(span_scores.dtype).min
        mask_value = torch.full([], mask_value, dtype=span_scores.dtype, device=span_scores.device)
        span_scores = torch.where(causal_mask, span_scores.to(span_scores.dtype), mask_value)

        # # hard evaluation
        # if self.lth_hard_inference:
        #     depth = 2
        #     max_bucket_score = torch.max(span_scores, dim=-1, keepdim=True).values
        #     thold_bucket_score = max_bucket_score - 2*depth
        #     qualifying = (span_scores >= thold_bucket_score).type(span_scores.dtype)
        #     span_scores = span_scores * qualifying

            #stats
            # for l, location in enumerate([32,64,128,256,512,1024]):
            #     num = qualifying.sum(dim=-1) # b,a,s
            #     stats[l][self.layer_idx] += num[:,:,(location-1)].mean(dim=0).cpu() # a
            # if self.layer_idx == 0:
            #     stats_count += 1
        if hard:
            #span_scores = ((span_scores - self.lth_thold) >= 0).float()
            pass # return the raw span scores for retreiver to decide how to use
        else:
            span_scores = nn.functional.sigmoid(span_scores - self.lth_thold)
        #span_scores = nn.functional.tanh(span_scores)
        return span_scores

