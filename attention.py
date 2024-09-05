import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,drop=0.5,qkvbias=False) -> None:
        super().__init__()
        assert (d_out%num_heads==0),"output dim should be divisible by number of heads"

        self.d_out=d_out

        self.w_query=nn.Linear(d_in,d_out,bias=qkvbias)
        self.w_key=nn.Linear(d_in,d_out,bias=qkvbias)
        self.w_value=nn.Linear(d_in,d_out,bias=qkvbias)

        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads

        self.drop=nn.Dropout(drop)
        
        #the last layer
        self.out_proj=nn.Linear(d_out,d_out,bias=qkvbias)

    
    def forward(self,x):
        batch,num_tokens,input_dim=x.shape

        queries=self.w_query(x)
        key=self.w_key(x)
        value=self.w_value(x)
        
        queries=queries.view(batch,num_tokens,self.num_heads,self.head_dim)
        key=key.view(batch,num_tokens,self.num_heads,self.head_dim)
        value=value.view(batch,num_tokens,self.num_heads,self.head_dim)

        #lets transpose 
        queries=queries.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)

        attention_score=queries@key.transpose(2,3)

        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]
        attention_score.masked_fill_(mask_bool,-torch.inf)
       
        attention_weight=torch.softmax(attention_score/key.shape[-1]**0.5,dim=-1)

        attention_weight=self.drop(attention_weight)

        context_vector=(attention_weight@value).transpose(1,2)

        context_vector=context_vector.contiguous().view(batch,num_tokens,self.d_out)
        context_vector=self.out_proj(context_vector)
        return context_vector