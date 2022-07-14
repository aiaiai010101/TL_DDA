# 使用 os.getcwd() 获取当前路径
from ASC.data_process.utils import *
import numpy as np

def data_process(config):

    ## 制作vocab_embedding和word_2_index
    vocab_embedding_file_path=config['vocab_embedding_file_path']
    word_dict, vocab_embedding = load_w2v(vocab_embedding_file_path,config['vocab_embedding_dim'] )
    # 保存 vocab_embedding和word_dict，供后边的网络模型使用
    vocab_embedding_value_path=config['vocab_embedding_value_path']
    np.save(vocab_embedding_value_path, vocab_embedding)
    np.save(config['word_dict_path'], word_dict)


    #制作其他训练所需数据
    original_train_file_path=config['original_train_file_path']
    original_test_file_path = config['original_test_file_path']

    train_data_path=config['train_data_path']
    test_data_path = config['test_data_path']

    make_train_or_test_data(original_train_file_path,train_data_path,word_dict,config['max_sentence_len'], config['max_target_len'] )
    make_train_or_test_data(original_test_file_path, test_data_path, word_dict, config['max_sentence_len'], config['max_target_len'])

