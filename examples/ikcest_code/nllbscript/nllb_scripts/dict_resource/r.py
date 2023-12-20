from fairseq.data import Dictionary
# dic = Dictionary.load("dict.joint.txt")
# print(len(dic))

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def is_contains_arabic(strs):
    for _char in strs:
        if '\u0600' <= _char <= '\u06ff':
            return True
    return False

file = "dict.txt"

def read(file):
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    return lines

# with open(file,"r",encoding='utf-8') as f:
#     lines = f.readlines()
#     res=[]
#     for l in lines:
#         word,freq = l.strip().rsplit(" ",1)
#         res.append(word)
#     print(len(res))
#     print(len(set(res)))
lines = read(file)
num_arb=0
num_zh=0
for line in lines:
    line = line.strip()
    if is_contains_arabic(line):
        num_arb+=1
        # print(line)
    if is_contains_chinese(line):
        num_zh+=1
        print(line)

# print(num_arb)
print(num_zh)


a={"你":1,"好":2}
keys = list(a.keys())
for k in keys:
    if k=="你":
        del a[k]
        a["我"] =1

# for k,v in a.items():
#     print(k,v)
#     if k=="你":
#         del a["你"]
print(a)
