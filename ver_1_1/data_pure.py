import re
import sys

def train_pure2(file, outfile):
    res = []
    reg = re.compile("<s>(.*?)</s>", re.IGNORECASE | re.DOTALL)
    with open(file, "r") as f:
        for line in f:
            cn_en = reg.findall(line)
            if len(cn_en) == 2:
                res.append(cn_en[1].strip() + '\t' + cn_en[0].strip())
    with open(outfile, 'w') as f:
        f.writelines("%s\n" % item for item in res)
    print('English to Chinese txt prepared with %s pairs.'%len(res))


if __name__ == "__main__":
    if len(sys.argv) < 2: print ('No input file!')
    else:
        path_to_file = 'corpus/en_cn.txt'
        train_pure2(sys.argv[1], path_to_file)