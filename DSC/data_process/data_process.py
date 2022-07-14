# 使用 os.getcwd() 获取当前路径
from DSC.data_process.utils import *
import numpy as np

def data_process(config):

    # 制作拼接后的sentiment_label_train
    # original_sentiment_label_train_path=config['original_sentiment_label_train_path']
    # new_sentence_label_file_path=config['new_sentence_label_file_path']
    # make_new_sentiment_label(original_sentiment_label_train_path,
    #                                                   new_sentence_label_file_path,config['max_sentence_len'])

    ## 制作vocab_embedding和word_2_index
    vocab_embedding_file_path=config['vocab_embedding_file_path']
    word_dict, vocab_embedding = load_w2v(vocab_embedding_file_path,config['vocab_embedding_dim'] )
    # 保存 vocab_embedding和word_dict，供后边的网络模型使用
    vocab_embedding_value_path=config['vocab_embedding_value_path']
    np.save(vocab_embedding_value_path, vocab_embedding)
    np.save(config['word_dict_path'], word_dict)


    #制作其他训练所需数据
    # new_sentence_label_file_path=config['new_sentence_label_file_path']
    # train_data_path=config['train_data_path']
    # make_train_data(new_sentence_label_file_path,train_data_path,word_dict,config['max_sentence_len'])


