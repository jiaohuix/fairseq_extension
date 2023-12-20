from zhconv import convert
# sent = "購入"
# sent = convert(sent, "zh-cn")
# print(sent)


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def read(file):
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def write(res,file):
    with open(file, "w", encoding='utf-8') as f:
        f.writelines(res)


# 两遍更新繁体字： 1.找到所有中文词，并清理错词  2.将中文中繁体词，先转简体，看看有没有，如果简体没有，就把繁体替换成简体，否则不动
# def vocab2_simple(lines):
#     # 1.clean err word, record  position of chinese
#     zh_pos = []
#     lines_new = []
#     for idx, line in enumerate(lines):
#         line = line.strip()
#         if len(line.split()) != 2:
#             line = f"{idx} 1"
#         if is_contains_chinese(line):
#             zh_pos.append(idx)
#         lines_new.append(line+"\n")
#
#     # 2.convert hant to hans
#     for zh_idx in zh_pos:
#         line = lines_new[zh_idx].strip()
#         simple_line =  convert(line,"zh-cn")
#         if simple_line not in lines_new: # 繁体，并且其简体未出现过, 时间复杂度太高了
#             lines_new[zh_idx] = simple_line+"\n"
#     return lines_new







def vocab2_simple(lines):
    # 1.clean err word, record  position of chinese
    zh2pos = {}
    for idx, line in enumerate(lines):
        line = line.strip()
        if len(line.split()) != 2:
            lines[idx] = f"{idx} 1\n"
        if is_contains_chinese(line):
            zh2pos[line] = idx

    # 2.convert hant to hans (hans not exists)
    zh_tokens = list(zh2pos.keys())
    for zh_token in zh_tokens:
        simple_token =  convert(zh_token,"zh-cn")
        if simple_token not in zh2pos.keys():
            zh_idx = zh2pos[zh_token]
            # update token
            lines[zh_idx] = simple_token+"\n"
            # update dict
            del zh2pos[zh_token]
            zh2pos[simple_token] = zh_idx

    return lines


file ="dict.txt"
lines = read(file)
# res=[]
# num=0
# dedup=set()
# for idx,line in enumerate(lines):
#     line = line.strip()
#     if len(line.split())!=2:
#         line = f"{idx} 1"
#     if is_contains_chinese(line):
#         if line not in dedup:
#             dedup.add(line)
#
#         new_line = convert(line,"zh-cn")
#         # if new_line=="国际 1":
#         #     print(new_line)
#         # 新的和老的不相等，并且和之前没重复  才能加入结果
#         if line!=new_line and (new_line not in dedup):
#             res.append(new_line+"\n")
#             num+=1
#             dedup.add(new_line)
#         else:
#             # res.append(line.replace(" 1","")+"  #fairseq:overwrite"+"\n")  # 先繁体，后简体会出现bug
#             line = convert(line,"zh-hant")
#             res.append(line+"\n")  # 先繁体，后简体会出现bug
#             dedup.add(line)
#
#         # if line!=new_line:
#         #     if new_line in dedup:
#         #         new_line+=" #fairseq:overwrite"
#                 # new_line = line
#             # else:
#             #     print(line,new_line)
#             #     num+=1
#         # res.append(new_line+"\n")
#     else:
#         res.append(line+"\n")


res = vocab2_simple(lines)
# print(num)
print(len(res))
write(res,"dict.nllb.simple.txt")