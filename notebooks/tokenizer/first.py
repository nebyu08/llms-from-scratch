import re


with open('verdict.txt',"r",encoding="utf-8") as f:
    txt=f.read()


temp=re.split(r'([,.]|\s)', txt)
#lets strip the white space
tokens=[item for item in temp if item.split()]

print(tokens)