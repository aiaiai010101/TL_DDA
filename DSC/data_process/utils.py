import random
import os
import numpy as np
## sentence_label中每个句子格式是： 前半部分情感 后半部分情感 前半部分长度 后半部分长度 句子内容
# 没有拼接的句子前半部分情感为 1 或者 -1，后半部分情感统一为0(在网络中不参与运算)，后半部分长度也为0
# 制作拼接句子后的数据
def make_new_sentiment_label(original_sentiment_label_train_path,new_sentence_label_file_path,max_sentence_len):
    print(1)
    fp = open(original_sentiment_label_train_path)
    # sentence[0]存储所有label为1的句子，sentence[1]存储所有label为0的句子
    # sentence_len存储句子长度
    sentence=[ [],[] ]
    sentence_len=[ [],[] ]
    for line in fp:
        line_sentence=line[2:]
        line_words = line_sentence.encode('utf8', 'ignore').decode('utf8', 'ignore').split()
        label=line[0]
        words_num = len(line_words)
        #扔掉长句子
        if words_num<=max_sentence_len:
            if eval(label)==1:
                sentence[0].append(line_sentence)
                sentence_len[0].append(words_num)
            else:
                sentence[1].append(line_sentence)
                sentence_len[1].append(words_num)
    fp.close()
##制作拼接的句子
    # sentence_label中每个句子格式是： 前半部分情感 后半部分情感 前半部分长度 后半部分长度 句子内容
    sentence_label=[ [],[],[],[] ]
    sample_num=min( len(sentence[0]) , len(sentence[1]) )
#制作label为1在前半部分，label为0在后半部分的句子
    index_1,index_2=list(range(sample_num  ) ),list(range(sample_num  ) )
    random.shuffle(index_1)
    random.shuffle(index_2)
    for i,j in zip(index_1,index_2):
        tmp_1,tmp_2=sentence[0][i],sentence[1][j]
        tmp_1_len,tmp_2_len=sentence_len[0][i],sentence_len[1][j]
        sentence_concat=tmp_1[:-1]+' , '+tmp_2[:]
        #两个句子中间我加了个逗号，所以句子总长度也要加1
        sentence_concat_len=tmp_1_len+1+tmp_2_len
        #扔掉长句子
        if sentence_concat_len<=max_sentence_len:
            #格式是 0.5 -0.5 len_1 len_2 sentence_len sentence
            tmp='0.5 -0.5 '+str(tmp_1_len)+' '+str(tmp_2_len)+' '+str(sentence_concat_len)+' '\
                +sentence_concat
            sentence_label[2].append(tmp)
# 制作label为0在前半部分，label为1在后半部分的句子
    index_1,index_2=list(range(sample_num  ) ),list(range(sample_num  ) )
    random.shuffle(index_1)
    random.shuffle(index_2)
    for i,j in zip(index_1,index_2):
        tmp_1,tmp_2=sentence[1][i],sentence[0][j]
        tmp_1_len,tmp_2_len=sentence_len[1][i],sentence_len[0][j]
        sentence_concat=tmp_1[:-1]+' , '+tmp_2[:]
        #两个句子中间我加了个逗号，所以句子总长度也要加1
        sentence_concat_len=tmp_1_len+1+tmp_2_len
        #扔掉长句子
        if sentence_concat_len<=max_sentence_len:
            #格式是 -0.5 0.5 len_1 len_2 sentence_len sentence
            tmp='-0.5 0.5 '+str(tmp_1_len)+' '+str(tmp_2_len)+' '+str(sentence_concat_len)+' '\
                +sentence_concat
            sentence_label[3].append(tmp)
# 制作label为1的句子
    for i in range( len(sentence[0]) ):
        tmp = '1 0 ' + str(sentence_len[0][i]) + ' ' + str(0) + ' ' + str(sentence_len[0][i]) + ' ' \
              + sentence[0][i]
        sentence_label[0].append(tmp)
# 制作label为0的句子
    for i in range( len(sentence[1]) ):
        tmp = '-1 0 ' + str(sentence_len[1][i]) + ' ' + str(0) + ' ' + str(sentence_len[1][i]) + ' ' \
              + sentence[1][i]
        sentence_label[1].append(tmp)
    #把所有的sentence_label整合成一个列表
    new_sentence_label_txt=sentence_label[0]+sentence_label[1]+sentence_label[2]+sentence_label[3]
    dir=os.path.dirname(original_sentiment_label_train_path)
    dir=os.path.abspath(dir)
    file_name=dir+'/new_sentence_label.txt'
    f = open(file_name, 'w')
    f.writelines(new_sentence_label_txt)
    f.close()

    return

#制作word_dict和vocab_embedding
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
    w2v = np.asarray(w2v)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt)) #把所有词向量的平均向量求出来，加到w2v最后一行
#    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)  #这是对应最后一个行向量

    return word_dict, w2v

def make_train_data(new_sentence_label_file_path,train_data_path,word_dict,max_sentence_len ):
    print(1)
    word_to_id = word_dict
    x, y, sen_1_mask, sen_2_mask, sen_mask, sen_len, = [], [], [], [], [], []
    f=open(new_sentence_label_file_path)
    lines =f.readlines()
    f.close()
    for i in range(len(lines)):
        # 在这里，strip是去除字符串结尾的\n
        line = lines[i].lower().strip().split()

        y.append( [eval(line[0]), eval(line[1])] )

        sen=line[5:]
        sen_id=[  word_to_id[ele] if ele in word_to_id else word_to_id['$t$'] for ele in sen ]
        x.append( sen_id+[0] * (max_sentence_len - len(sen_id)))

        tmp_sen_1_len,tmp_sen_2_len=eval(line[2]), eval(line[3])
        tmp_sen_1_mask=[1]*tmp_sen_1_len+[0]*(max_sentence_len-tmp_sen_1_len)
        #注意我在拼接句子中间加的','
        tmp_sen_2_mask=[0]*tmp_sen_1_len+[0]+[1]*tmp_sen_2_len+\
                       [0]*(max_sentence_len-tmp_sen_1_len-1-tmp_sen_2_len)

        sen_1_mask.append(tmp_sen_1_mask)
        sen_2_mask.append(tmp_sen_2_mask)

        # sen_mask和sen_len可以在运算中快速地由x得到，这里就不存储了，以免占内存
        # tmp_sen_len=eval(line[4])
        # tmp_sen_mask=[1]*tmp_sen_len+[0]*(max_sentence_len-tmp_sen_len)
        # sen_mask.append(tmp_sen_mask)
        # sen_len.append(tmp_sen_len)

    x=np.asarray(x,dtype=np.int32)
    y=np.asarray(y,dtype=np.float32)
    sen_1_mask=np.asarray(sen_1_mask,dtype=np.bool_)
    sen_2_mask=np.asarray(sen_2_mask,dtype=np.bool_)

#    sen_mask=np.asarray(sen_mask)
#    sen_len=np.asarray(sen_len)
    np.savez(train_data_path, sentence=x, label=y, sen_1_mask=sen_1_mask, sen_2_mask=sen_2_mask)

    return









