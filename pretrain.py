import torch 
import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence


class GptDataSetv1(Dataset):
    def __init__(self,tokenizer,dataset,context_length,stride) -> None:
        super().__init__()
        self.tokenizer=tokenizer

        #lets tokenize the text
        self.tokens=self.tokenizer.encode(dataset,allowed_special={"<|endoftext|>"})   #array of ids
        
        self.inputs=[]
        self.outputs=[]

        for i in range(0,len(self.tokens),stride):
            input_chunks=self.tokens[i:i+context_length]
            output_chunks=self.tokens[i+1:i+context_length+1]

            #lets append
            if(len(input_chunks)==context_length and len(output_chunks)==context_length):
                self.inputs.append(torch.tensor(input_chunks))
                self.outputs.append(torch.tensor(output_chunks))
    
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index) :
        #purpose of this function is to make an input and output matcher
        return self.inputs[index].clone().detach(),self.outputs[index].clone().detach()
    

def collate_fn(batch):
    inputs,outputs=zip(*batch)
    inputs=pad_sequence(inputs,batch_first=True,padding_value=0)
    outputs=pad_sequence(outputs,batch_first=True,padding_value=0)
    return inputs,outputs

def create_dataloader_v1(txt,batch_size=4,context_length=120,stride=128,shuffle=True,drop_last=True):
    tokenizer=tiktoken.get_encoding('gpt2')
    dataset=GptDataSetv1(tokenizer,txt,context_length,stride)
    #prepare the datalaoder
    dataloader=DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          drop_last=drop_last
                         )
    return dataloader

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

#loss

#for calculating the loss of a single batch
def loss_batch(inputs,target,model,device):
    #lets move all varaible into the same device
    inputs,target=inputs.to(device),target.to(device)
    
    logits=model(inputs)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
    return loss

#lets calculate the loss for the whole batch
def total_loss_batches(dataloader,model,device,num_batches=None):
    if(num_batches==None):
        num_batches=len(dataloader)
    else:
        num_batches=min(num_batches,dataloader)
    
    #lets calculate the loss over batches
    total_loss=0
    for i,(inputs,target) in enumerate(dataloader):
        if(i<num_batches):
            loss=loss_batch(inputs,target,model,device)
            total_loss+=loss.item()
        else:
            break
    
    total_loss=total_loss/num_batches
    return total_loss


#generate new tokens
def generate_new_tokens(model,device,start_context,tokenizer,max_tokens):
    model.eval()
    ids=text_to_ids(start_context,tokenizer).to(device)
    context_length=model.pos_emb.weight.shape[0]
    new_ids=generate_text(model,ids,context_length,max_tokens)
    
    #convert idx into text
    with torch.no_grad():
        new_text=ids_to_text(new_ids,tokenizer)
        
    model.train()
    print(f"\n {new_text}")


def eval_mode(model,train_loader,val_loader,device,eval_batch):
    model.eval()
    with torch.no_grad():
        train_loss=total_loss_batches(train_loader,device,num_batches=eval_batch)
        eval_loss=total_loss_batches(val_loader,device,num_batches=eval_batch)
    model.train()
    return train_loss,eval_loss



def model_trainer(train_dataloader,val_dataloader,device,train_epoch,eval_freq,eval_batch,val_epoch,model,start_context,max_tokens,tokenizer,optimizer):
    #for trainig
    num_tokens_seen=0
    #tracking the tokens
    track_tokens_seen=[]
    
    #eval_tokens_seen=[]
    train_losses=[]
    eval_losses=[]
    
    for epoch in range(train_epoch):
        model.train()
        for i,(inputs,targets) in enumerate(train_dataloader):
            #train the model

            optimizer.zero_grad()

            #calculate the loss
            loss=loss_batch(inputs,targets,model,device)

            #backpropagation
            loss.backward()

            #model update
            optimizer.step()

           
            num_tokens_seen+=inputs.numel()

            if(i%eval_freq==0):
                #for evaluation
                train_loss,eval_loss=eval_mode(model,train_dataloader,val_dataloader,eval_batch)

                #for recording
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                #the tokens
                track_tokens_seen.append(num_tokens_seen)

                print(f"for epoch: {epoch}:iteration {i}: train loss {train_losses}: eval loss {eval_losses}")
            
        
        #lets generate the new tokens
        generate_new_tokens(model,device,start_context,tokenizer,max_tokens)
                
    return train_losses,eval_losses,track_tokens_seen


#EXTRAS
def text_to_ids(text,tokenizer):
    #this convert text into token ids
    
    encoded=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor=torch.tensor(encoded)
    encoded_tensor=encoded_tensor.unsqueeze(dim=0)
    return encoded_tensor

def ids_to_text(ids,tokenizer):
    #this converts the tokens ids into text
    return tokenizer.decode(ids.squeeze(dim=0).tolist())


#for newer generating 
def latest_generate(model,idx,context_length,new_token_length,device,temprature,topk):
    idx_cont=idx[:,-context_length:] #2d inputs num of tokens by embeding dim
    for _ in range(new_token_length):
        with torch.no_grad():
            logits=model(idx_cont)

        #lets apply topk
        logits=logits[:, -1, :]  #take only last tokens prediction
        if(topk is not None):
            top_logits,_=torch.topk(logits,k=topk)
            min_value=top_logits[:,-1]
            
            logits=torch.where(
                logits<min_value.unsqueeze(dim=-1),
                torch.tensor(float('-inf')).to(device),
                logits
            )
        
        #lets apply multinomial
        if(temprature>0.0):
            logits=logits/temprature
            probs=torch.softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,num_samples=1)
        else:
            probs=torch.softmax(logits,dim=-1)
            next_token=torch.argamax(probs,dim=-1,keepdim=True)
            
        idx=torch.cat((idx,next_token),dim=1)
    
    return idx