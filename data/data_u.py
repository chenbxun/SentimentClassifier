import os
import pickle
import csv
import thulac
import re
import json

DATA_PATH = './usual'
id2tag = ['neutral', 'happy', 'angry', 'sad', 'fear', 'surprise']
tag2id = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4, 'surprise': 5}
word2id = {}
id2word = []
count = {}

fileters = ['"', '＂', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', '：', ';', '；', '<', '=', '>', '·', '\'', '’',
                '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '\t', '\n', '\x97', '\x96', '”', '“', '《', '》', '〈', '〉', '／',
                '\【', '\】', '\（', '\、', '。', '\）',  '\，', ']', '[', '【', '】', '——', '→', '—', '‘', '^', '⋯⋯', '．', '「',
                '[\sa-zA-Z]', '-?(\d+(\.\d*)?|\.\d+)']
#…、？、！、～这类词能够传达出情感
thu1 = thulac.thulac(seg_only=True)

def handle_data(data_type):
    sentences = []
    labels = []

    is_first = True
    with open(os.path.join(DATA_PATH, data_type + ".csv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader: # 对每一行文本
            if is_first:
                is_first = False
                continue

            if row[1] == "":
                continue
            
            # 清洗数据
            sentence = row[1].strip()
            sentence = re.sub("|".join(fileters), " ", sentence) # 替换特殊符号
            sentence = thu1.cut(sentence, text=True).split(" ") # 分词

            #统计全局词频
            for word in sentence:
                if word not in count:
                    count[word] = 1
                else:
                    count[word] += 1

            sentences.append(sentence)
            labels.append(tag2id[row[2]])
    
    return sentences, labels

def save_count(path):
    sorted_count = sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))
    # 将排序后的词频写入CSV文件
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'frequency'])  # 写入表头
        for word, freq in sorted_count:
            writer.writerow([word, freq])

if __name__ == "__main__":
    min = 2
    x_train, y_train = handle_data("train")
    x_val, y_val = handle_data("val")

    save_count('./usual/word_frequencies.csv')
    # 删除低频词
    if min is not None:
        count = {word: value for word, value in count.items() if value >= min}

    # 建立词语到id的映射
    for word in count:
        if word not in word2id:
            word2id[word] = len(word2id)
            id2word.append(word)
    id2word.append('<unk>') # 对于未登录词，使用id=len(word2id)

    # 将句子中的词语替换为id
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            if x_train[i][j] in word2id:
                x_train[i][j] = word2id[x_train[i][j]]
            else:
                x_train[i][j] = len(word2id)
    
    for i in range(len(x_val)):
        for j in range(len(x_val[i])):
            if x_val[i][j] in word2id:
                x_val[i][j] = word2id[x_val[i][j]]
            else:
                x_val[i][j] = len(word2id)

    print(x_train[0])
    print([id2word[i] for i in x_train[0]])
    print(y_train[0])
    print(x_val[0])
    print([id2word[i] for i in x_val[0]])
    print("len(word2id): ", len(word2id))
    
    with open("./datasave.pkl", 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_val, outp)
        pickle.dump(y_val, outp)

