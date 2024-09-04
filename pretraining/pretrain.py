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