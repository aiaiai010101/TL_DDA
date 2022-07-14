import random
import os
import numpy as np
import spacy
import string

all_pos_ = ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON',
            'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE', 'SYM')
#value从1开始，0是留给填充值的
pos_dict={'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10,
          'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16,  'X': 17, 'SPACE':18, 'SYM':19}

def load_w2v(w2v_file, embedding_dim ):
    fp = open(w2v_file)
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    # a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for line in fp:
        line = line.encode('utf8', 'ignore').decode('utf8', 'ignore').split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print(u'a bad word embedding: {}'.format(line[0]))
            continue
        cnt += 1
        # if line[0] ==
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt)) #把所有词向量的平均向量求出来，加到w2v最后一行
#    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)  #这是对应最后一个行向量

    return word_dict, w2v


def change_y_to_onehot(y):
    from collections import Counter
    # print(Counter(y))
    class_set = set(y)  #set函数结果有时会变化，还是固定好
    n_class = len(class_set)
    # y_onehot_mapping = dict(zip(class_set, range(n_class)))
    y_onehot_mapping={'1': 0, '0': 1, '-1': 2}  #固定比较好，防止变动
    # print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def get_adjacency_matrix(adjacency):
    adjacency_matrix = np.zeros([len(adjacency), len(adjacency)])
    for i in range(len(adjacency)):
        c_adjacency=adjacency[i]
        if c_adjacency[0]!=c_adjacency[1]:
            adjacency_matrix[ c_adjacency[0], c_adjacency[1] ]=1
            adjacency_matrix[c_adjacency[1], c_adjacency[0]] = 1

    return adjacency_matrix



def get_syntax_length(adjacency_matrix,TARGET_INDEX ):
    c_target_index=TARGET_INDEX
    is_visited=[0]*adjacency_matrix.shape[0]
    is_visited[c_target_index]=1
    distance=[0]*adjacency_matrix.shape[0]
    distance[c_target_index] = 0
    queue=[c_target_index]
    while(len(queue)!=0):
        now=queue[0]
        c_adjacency=adjacency_matrix[now]
        c_adjacency_index=np.argwhere( c_adjacency==1 )
        # c_adjacency_index是个二维矩阵，转为一维
        c_adjacency_index = c_adjacency_index.flatten()
        for i in c_adjacency_index:
            if is_visited[i]==0:
                distance[i]=distance[now]+1
                is_visited[i]=1
                queue.append(i)
        del queue[0]
    syntax_length=distance
    syntax_length=  np.array(syntax_length )
    max_syntax_length=syntax_length.max()
    # 还有一些没法到达的单词，距离要设置，不能为0
    for i in range(len(syntax_length)):
        if i != TARGET_INDEX and syntax_length[i]==0:
            syntax_length[i]=max_syntax_length+1

    return syntax_length,max_syntax_length




def make_train_or_test_data(original_file_path,data_path,word_dict,max_sentence_len, max_target_len ):
    nlp = spacy.load('en_core_web_sm')

    word_to_id = word_dict
    x, sen_len, target_word, target_len, y, position, attention_1, pos, syntax_position = [], [], [], [], [], [], [], [], []
    f=open(original_file_path)
    lines =f.readlines()
    f.close()

    encoding = 'utf8'
    for i in range(0, len(lines), 4):
        # target
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        current_target_word=[]
        for w in words_2:
            if w in word_to_id:
                current_target_word.append(word_to_id[w])
            else:
                current_target_word.append(word_to_id['$t$'])
        l = min(len(current_target_word), max_target_len)  #限制target的长度，不得超过10
        target_len.append(l)
        target_word.append(current_target_word[:l] + [0] * (max_target_len - l))

        # y
        y.append(lines[i + 1].strip().split()[0])

        # sentence
        words_1 = lines[i + 2].encode(encoding).decode(encoding).lower().split()
        words, pp = [], []
        for word in words_1:
            t = word.split('/')
            ind = int(t[-1])
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
            else:
                words.append(word_to_id['$t$'])
            pp.append(ind)

        words = words[:max_sentence_len] #限制句子长度
        sen_len.append(len(words))
        pp = pp[:max_sentence_len]
        x.append(words + [0] * (max_sentence_len - len(words)))
        position.append(pp + [0] * (max_sentence_len - len(pp)))

        # pos and syntax_position
        words_3 = lines[i + 2].encode(encoding).decode(encoding).lower().split()
        word_3 = [ele.split('/')[0] for ele in words_3]
        idx = [ele.split('/')[-1] for ele in words_3]
        TARGET_INDEX = int(idx[0])
        target = lines[i]
        target = target.strip().lower().split()
        full_sentence= word_3[:TARGET_INDEX]  + ['target'] + word_3[TARGET_INDEX:]
        #对full_sentence要做特殊处理，因为spacy的分词方式和训练文件中不一样，不处理会对不上号
        #处理方式就是把所有带标点的单词全部删去标点符号,除去最后一个
        for index in range(len(full_sentence)-1):
            if len(full_sentence[index])>1 and full_sentence[index]!='/' and full_sentence[index]!=',' and full_sentence[index]!='.':
                full_sentence[index]=full_sentence[index].translate(str.maketrans('', '', string.punctuation))
        full_sentence=' '.join(full_sentence)
        doc = nlp(full_sentence)

        # m=0
        # for token in doc:
        #     m=m+1
        #     print(token, token.pos_)

        c_pos=[  pos_dict[token.pos_]   for token in doc  ]
        del c_pos[TARGET_INDEX]
        #邻接关系
        adjacency=[ [token.i, token.head.i]  for token in doc  ]
        #构建邻接矩阵
        adjacency_matrix=get_adjacency_matrix(adjacency)
        #计算各单词到target的最短语义距离
        syntax_length,max_syntax_length=get_syntax_length(adjacency_matrix,TARGET_INDEX )
        syntax_length=syntax_length.tolist()
        #为了防止结尾的标点符号干扰，把它们的语义距离统一设置为max_syntax_length+1
        if nlp(word_3[-1])[0].is_punct:
            syntax_length[-1]=max_syntax_length+1
        del syntax_length[TARGET_INDEX]
        #pad
        c_pos=c_pos[:max_sentence_len]
        syntax_length=syntax_length[:max_sentence_len]
        pos.append(c_pos + [0] * (max_sentence_len - len(c_pos)))
        syntax_position.append(syntax_length + [0] * (max_sentence_len - len(syntax_length)))

        # attention
        attention = []
        attention_words = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for atten in attention_words:
            attention.append(atten)
        attention = attention[:max_sentence_len]
        attention=[float(ele) for ele in attention]
        attention_1.append(attention + [0] * (max_sentence_len - len(attention)))

    y = change_y_to_onehot(y)

    attention_1 = np.asarray(attention_1, dtype=np.float32)
    target_word= np.asarray(target_word, dtype=np.int32)

    pos = np.asarray(pos, dtype=np.int32)
    syntax_position = np.asarray(syntax_position, dtype=np.int32)

    #     x, sen_len, target_word, target_len, y, position, attention_1
    np.savez(data_path, sentence=x, label=y, sen_len=sen_len, target_word=target_word,
             target_len=target_len, position=position, attention_1=attention_1, pos=pos, syntax_position=syntax_position)

    return









