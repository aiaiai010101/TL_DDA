import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

class ASC_train_network(nn.Module):
    def __init__(self, dropout, vocab_embedding, position_embedding_dim,pos_embedding_dim, max_sentence_len,  bilstm_embed_size, bilstm_num_layers,
                 bidirectional,MIN_VALUE):
        super(ASC_train_network, self).__init__()

        self.attention_weight=nn.Parameter( torch.Tensor( 2*bilstm_embed_size,2*bilstm_embed_size ) )
        self.attention_fusion_weight = nn.Parameter(torch.Tensor(max_sentence_len, max_sentence_len))
        self.vocab_embedding=vocab_embedding
        self.position_embedding=nn.Embedding(num_embeddings=max_sentence_len,
                                   embedding_dim=position_embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=20,
                                               embedding_dim=pos_embedding_dim)
        # self.syntax_position_embedding = nn.Embedding(num_embeddings=80,
        #                                         embedding_dim=pos_embedding_dim)
        self.target= nn.Parameter(torch.Tensor(2*bilstm_embed_size))

        self.dropout=dropout
        self.MIN_VALUE=MIN_VALUE
        self.rnn_1 = nn.GRU(
            input_size=vocab_embedding.weight.shape[1]+position_embedding_dim+pos_embedding_dim,
            hidden_size=bilstm_embed_size,   #这个参数决定了隐层的维度，其实也决定了输出的维度，输出embed维度=hidden_size*(2 if bidirectional==True else 1)
            num_layers=bilstm_num_layers,   #这个参数是GRU里隐层的层数，默认为1。这个值的设置与bidirectional无关，请注意这一点。
            bidirectional=bidirectional,
            batch_first=True
        )#True则输入输出的数据格式为 (batch, seq, feature)
        self.rnn_2 = nn.GRU(
            input_size=vocab_embedding.weight.shape[1],
            hidden_size=self.rnn_1.hidden_size,   #这个参数决定了隐层的维度，其实也决定了输出的维度，输出embed维度=hidden_size*(2 if bidirectional==True else 1)
            num_layers=bilstm_num_layers,   #这个参数是GRU里隐层的层数，默认为1。这个值的设置与bidirectional无关，请注意这一点。
            bidirectional=bidirectional,
            batch_first=True
        )#True则输入输出的数据格式为 (batch, seq, feature)
        self.classification_transform = nn.Sequential(
            nn.Linear( bilstm_embed_size*2 , 3),
            nn.Dropout(dropout)
        )
        # self.classification_transform_2 = nn.Parameter( torch.Tensor( 2*bilstm_embed_size,1 ) )

        self._reset_parameters()


    def _reset_parameters(self):
        random_base = 0.1
        init.uniform_(self.position_embedding.weight, -random_base, random_base)
        init.uniform_(self.pos_embedding.weight, -random_base, random_base)
        init.uniform_(self.attention_weight, -random_base, random_base)
        init.uniform_(self.attention_fusion_weight, -random_base, random_base)

        # init.xavier_uniform_(self.position_embedding.weight)
        # init.xavier_uniform_(self.pos_embedding.weight)
        # init.xavier_uniform_(self.attention_weight)
        # init.xavier_uniform_(self.attention_fusion_weight)



    def forward(self, sentence, sen_len, target_word, target_len, position, attention_1, pos, syntax_position):
        ##先做数据转换，pytorch要求indice必须为int64类型
        sentence = sentence.long()
        position=position.long()
        pos = pos.long()
        target_word=target_word.long()

        ## word_embedding和position_embedding和pos_embedding
        # word_embedding.shape:[batch_size,max_sen_len,300]
        sentence_embedding=self.vocab_embedding(sentence)
        # position_embedding.shape:[batch_size,max_sen_len,100]
        position_embedding = self.position_embedding(position)
        # pos_embedding.shape:[batch_size,max_sen_len,100]
        pos_embedding=self.pos_embedding(pos)
        # # syntax_position_embedding.shape:[batch_size,max_sen_len,100]
        # syntax_position_embedding=self.syntax_position_embedding(syntax_position)

        # position_embedding.shape:[batch_size,max_sen_len,500]
        inputs_s=torch.cat((sentence_embedding, position_embedding,pos_embedding), 2)

        ## target
        target_embedding=self.vocab_embedding(target_word)

        ##bilstm
        # sentence 和 target的hidden
        inputs_s = F.dropout(inputs_s, p=self.dropout, training=self.training)
        hiddens_s,_ =self.rnn_1(inputs_s)

        # target
        target = F.dropout(target_embedding, p=self.dropout, training=self.training)
        hiddens_t, _ = self.rnn_2(target)

        # 计算出每个句子所有hiddens_t(不包括pad部分)的均值，作为aspect term的初始表达
        target_mask=(target_word != 0)
        target_mask=target_mask.unsqueeze(-1)
        pool_t_1 =  ( target_mask*hiddens_t ).sum( axis=1 ) / target_mask.sum( axis=1 )


        # pool_t_1=self.target.repeat(sentence.shape[0], 1)

        ## 计算aspect term对句子中各单词的注意力
        pool_t_1=pool_t_1.unsqueeze(1)
        pool_t_1_T = pool_t_1.reshape( pool_t_1.shape[0], pool_t_1.shape[2], pool_t_1.shape[1]   )
        tmp = hiddens_s.matmul(self.attention_weight)
        score = tmp.matmul(pool_t_1_T)
        score = score.squeeze(-1)

        sen_pad_mask = (sentence == 0)
        score = score.masked_fill(sen_pad_mask, self.MIN_VALUE)
        word_attention = F.softmax(score, dim=1)
        # word_attention.shape=[batch_size,max_sen_len]

        ## 融合attention_1(DSC得到) 和 word_attention(ASC得到)
        #校正attention
        # 此处调整方式与论文中所写的公式不同，没有用指数，我猜测是因为指数会导致不同距离的权重变化过快。
        # 此处调整方式是 1-(位置/句子长度) 得到每个位置的的权重，以此权重来提升近距离的位置注意力，削弱远距离位置注意力
        # 这里不用担心position中的pad影响，因为pad位置对应的attention_1的值也是0，调整后pad对应的attention_2的值仍然是0
        # 权重调整方法1   当前效果最好
        # sen_len = sen_len.unsqueeze(-1)
        # u_t = position / sen_len
        # w_t = 1.0 - u_t

        # 权重调整方法2
        # u_t = 2**(syntax_position-1)
        # w_t = 1.0 / (u_t+0.0000000001)

        # 权重调整方法3
        # w_t = 1.0 / (position + 0.0000000001)

        # 权重调整方法4 按照syntax_position来调整
        syntax_position_len=syntax_position.max(axis=1)[0]
        syntax_position_len = syntax_position_len.unsqueeze(-1)
        # 防止syntax_position_len过小
        u_t = syntax_position / (syntax_position_len+1)
        w_t = (1.0 - u_t)

        #把没有语法联系的单词权重设为极小值
        # syntax_position_no_link_mask = (syntax_position == syntax_position_len)
        # w_t = w_t.masked_fill( syntax_position_no_link_mask, 0.00000001)

        # 权重调整方法5 按照(syntax_position+position)/2来调整
        #####


        attention_2 = w_t * attention_1
        # 权重*attention1之后还要做个归一化，以保证调整后的注意力和为1
        attention_2 = attention_2 / attention_2.sum(axis=1,keepdim=True)


        gate=0.5
        att = (1 - gate) * word_attention + gate * attention_2
        ## 计算每个target的句子级表达
        att =att.unsqueeze(1)


        outputs_s_1 = att.matmul(hiddens_s)
        # 残差连接得到最终每个target的表达
        # pool_t_2 = outputs_s_1 + pool_t_1
        pool_t_2=outputs_s_1

        pool_t_2=pool_t_2.squeeze(1)
        target_sentiment = self.classification_transform(pool_t_2)


        # ##计算每个单词的情感极性
        # word_sentiment=hiddens_s.matmul(self.classification_transform_2)
        # word_sentiment=torch.sigmoid(word_sentiment)
        # word_sentiment = (word_sentiment - 0.5) * 2
        # ##计算整个句子的情感极性
        # target_sentiment=att.matmul(  word_sentiment  )
        # target_sentiment=target_sentiment.squeeze(-1)
        # target_sentiment = target_sentiment.squeeze(-1)



        return target_sentiment










