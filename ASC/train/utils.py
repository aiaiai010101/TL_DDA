from ASC.train.dataset import ASC_Dataset
from torch.utils.data import DataLoader
from ASC.train.network import *
import numpy as np
lamda = 0.1

def make_data_loader(config,data_source):

    train_data_path=config[data_source]['train_data_path']
    test_data_path=config[data_source]['test_data_path']
    train_data = ASC_Dataset(train_data_path)
    test_data = ASC_Dataset(test_data_path)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['net_parameter']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config['net_parameter']['test_batch_size'],
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader

def make_model(config,data_source):
    vocab_embedding_value_path=config[data_source]['vocab_embedding_value_path']
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
    position_embedding_dim=net_parameter['position_embedding_dim']
    pos_embedding_dim=net_parameter['pos_embedding_dim']

    max_sentence_len=config[data_source]['max_sentence_len']

    model=ASC_train_network(dropout, vocab_embedding, position_embedding_dim, pos_embedding_dim, max_sentence_len, bilstm_embed_size,
                            bilstm_num_layers,bidirectional,MIN_VALUE)

    return model

def get_accuracy(logit,label_scalar):
    pred = logit.argmax(dim=1)
    # label_scalar = label.argmax(dim=1)
    correct_samples=( pred==label_scalar  ).sum()
    batch_samples=logit.shape[0]
    accuracy = correct_samples / float(batch_samples)
    return accuracy


def get_accuracy_2(logit,train_label_scalar):
    pred = logit - logit
    idx = logit >= lamda
    pred = pred.masked_fill(idx, 1)
    idx = logit <= -lamda
    pred = pred.masked_fill(idx, -1)
    # idx= (logit>-lamda) * (logit<lamda)
    # pred = pred.masked_fill(idx, 0)


    correct_samples=( pred==train_label_scalar  ).sum()
    batch_samples=logit.shape[0]
    accuracy = correct_samples / float(batch_samples)
    return accuracy




def eval(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_samples = 0
        correct_samples = 0
        val_loss=0
        for i,data in enumerate(data_loader):
            test_sentence, test_label, test_sen_len, test_target_word, \
            test_target_len, test_position, test_attention_1, test_pos, test_syntax_position =data

            test_sentence, test_label, test_sen_len, test_target_word, \
            test_target_len, test_position, test_attention_1, test_pos, test_syntax_position = \
                test_sentence.to(device), test_label.to(device), test_sen_len.to(device), test_target_word.to(device), \
                test_target_len.to(device), test_position.to(device), test_attention_1.to(device), test_pos.to(device), \
                test_syntax_position.to(device)

            logit = model( test_sentence, test_sen_len, test_target_word, test_target_len, test_position,
                           test_attention_1, test_pos, test_syntax_position)
            test_label_scalar = torch.max(test_label, 1)[1]
            loss = criterion(logit, test_label_scalar)
            val_loss=val_loss+loss
            pred = logit.argmax(dim=1)
            correct_samples=correct_samples+(pred==test_label_scalar).sum()
            total_samples=total_samples+test_sentence.shape[0]
#        print(total_samples)
        accuracy = correct_samples / float( total_samples )
        avg_loss = val_loss / (i+1)
    return accuracy,avg_loss

def eval_2(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_samples = 0
        correct_samples = 0
        val_loss=0
        for i,data in enumerate(data_loader):
            test_sentence, test_label, test_sen_len, test_target_word, \
            test_target_len, test_position, test_attention_1, test_pos, test_syntax_position =data

            test_sentence, test_label, test_sen_len, test_target_word, \
            test_target_len, test_position, test_attention_1, test_pos, test_syntax_position = \
                test_sentence.to(device), test_label.to(device), test_sen_len.to(device), test_target_word.to(device), \
                test_target_len.to(device), test_position.to(device), test_attention_1.to(device), test_pos.to(device), \
                test_syntax_position.to(device)

            logit = model( test_sentence, test_sen_len, test_target_word, test_target_len, test_position,
                           test_attention_1, test_pos, test_syntax_position)
            test_label_scalar = torch.max(test_label, 1)[1]

            test_label_scalar = (test_label_scalar - 1) * (-1.)

            loss = criterion(logit, test_label_scalar)
            val_loss=val_loss+loss


            pred=logit-logit
            idx=logit>=lamda
            pred=pred.masked_fill(idx, 1)
            idx = logit <= -lamda
            pred = pred.masked_fill(idx, -1)
            # idx= (logit>-lamda) * (logit<lamda)
            # pred = pred.masked_fill(idx, 0)

            correct_samples=correct_samples+(pred==test_label_scalar).sum()
            total_samples=total_samples+test_sentence.shape[0]
#        print(total_samples)
        accuracy = correct_samples / float( total_samples )
        avg_loss = val_loss / i
    return accuracy,avg_loss