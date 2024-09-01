import string

vocab={letter:index for index,letter in enumerate(string.ascii_lowercase)}

with open('verdict.txt','r',encoding='utf-8') as f:
    text=f.read()

#lets merge and create a new one


#make the bpe from scratch
#add the embedding layer from sratch