import torch
import tiktoken
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,emb_dim) -> None:
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        mean=x.mean(keepdim=True,dim=-1)
        var=x.var(keepdim=True,dim=-1,unbiased=False)
        norm_value=(x-mean)/torch.sqrt(self.eps+var)  
        return self.scale*norm_value+self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2)/torch.tensor(torch.pi))*(x+0.044715*torch.pow(x,3))))
    

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg['emb_dim'],4*cfg['emb_dim']),
            GELU(),
            nn.Linear(4*cfg['emb_dim'],cfg['emb_dim'])
        )
    def forward(self,x):
        return self.layers(x)
    

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

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layer_norm1=LayerNorm(cfg['emb_dim'])
        self.layer_norm2=LayerNorm(cfg['emb_dim'])

        self.attention=MultiHeadAttention(d_in=cfg['emb_dim'],
                                      d_out=cfg['emb_dim'],
                                      context_length=cfg['context_length'],
                                      num_heads=cfg['n_heads'],
                                      drop=cfg['drop_rate'],
                                      qkvbias=cfg['qkv_bias']) 
        
        self.drop_residual=nn.Dropout(cfg['drop_rate'])
        self.feedforward=FeedForward(cfg)

    def forward(self,x):
        #first block
        residual=x  #residual attention
        x=self.layer_norm1(x)
        x=self.attention(x)
        x=self.drop_residual(x)

        #lets connect to residual
        x=x+residual

        #second block
        residual=x
        x=self.layer_norm2(x)
        x=self.feedforward(x)
        x=self.drop_residual(x)
        x=x+residual

        return x
    
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb=nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb=nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop=nn.Dropout(cfg['drop_rate'])
        self.transformer_block=nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        self.last_norm=LayerNorm(cfg['emb_dim'])
        self.out_prog=nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias=False)

    def forward(self,x):
        #shape
        batch,seq_length=x.shape

        toke_emb=self.token_emb(x)
        pos_emb=self.pos_emb(torch.arange(seq_length,device=x.device))
        
        x=toke_emb+pos_emb
        x=self.drop(x)

        x=self.transformer_block(x)
        x=self.last_norm(x)

        logits=self.out_prog(x)
        return logits


def generate_text(model,idx,context_length,new_token):
    for _ in range(new_token):
        idx=idx[:,-context_length:]
        with torch.no_grad():
            logits=model(idx)
            
        logits=logits[:,-1,:] #last token
        probs=torch.softmax(logits,dim=-1)
        next_word=torch.argmax(probs,dim=-1,keepdim=True)  #token position
        idx=torch.cat((idx,next_word),dim=1)
    return idx