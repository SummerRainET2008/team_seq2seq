import re

def train_pure2(file, outfile):
    res = []
    reg_cn = re.compile("'ch': ' <s>(.*?)</s>", re.IGNORECASE | re.DOTALL)
    reg_en = re.compile("'en': ' <s>(.*?)</s>", re.IGNORECASE | re.DOTALL)
    with open(file, "r") as f:
        for line in f:
            cn = reg_cn.findall(line)
            en = reg_en.findall(line)
            if len(cn) == len(en) == 1:
                res.append(en[0].strip() + '\t' + cn[0].strip())
    with open(outfile, 'w') as f:
        f.writelines("%s\n" % item for item in res)
    print('English to Chinese txt prepared with %s pairs.'%len(res))


if __name__ == "__main__":
    original_path = 'corpus/data002.zh_en.2.4M.pydict'
    path_to_file = 'corpus/en_cn.txt'
    train_pure2(original_path, path_to_file)
'''
Traceback (most recent call last):
  File "<input>", line 1, in <module>
  File "<input>", line 8, in train_pure2
UnicodeDecodeError: 'gbk' codec can't decode byte 0x8b in position 447: illegal multibyte sequence
# but no error on 201
'''