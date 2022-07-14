from DSC.train.dataset import DSC_Dataset
from torch.utils.data import DataLoader
from DSC.train.network import *
import numpy as np

def make_data_loader(config):

    train_data_path=config['train_data_path']
    train_data = DSC_Dataset(train_data_path)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['net_parameter']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    return train_loader

def make_model(config):
    vocab_embedding_value_path=config['vocab_embedding_value_path']
    vocab_embedding_value=np.load(vocab_embedding_value_path)
    #制作embedding
    vocab_embedding = nn.Embedding(num_embeddings=vocab_embedding_value.shape[0],
                                   embedding_dim=vocab_embedding_value.shape[1])
    # 初始化embedding层，用vocab_embedding_value
    vocab_embedding.weight.data.copy_(torch.tensor(vocab_embedding_value))
    # 冻结vocab_embedding的参数，不让其参加训练
    for name,param in vocab_embedding.named_parameters():
        param.requires_grad=False

    #制作网络模型
    net_parameter=config['net_parameter']
    dropout=net_parameter['dropout']
    bilstm_embed_size=net_parameter['bilstm_embed_size']
    bilstm_num_layers=net_parameter['bilstm_num_layers']
    bidirectional=net_parameter['bidirectional']
    MIN_VALUE=net_parameter['MIN_VALUE']
    model=DSC_train_network(dropout, vocab_embedding, bilstm_embed_size,
                            bilstm_num_layers,bidirectional,MIN_VALUE)

    return model