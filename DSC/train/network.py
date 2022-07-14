import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class attention_sentiment_score_network(nn.Module):
    def __init__(self, dropout, vocab_embedding, bilstm_embed_size, bilstm_num_layers,
                 bidirectional,MIN_VALUE):
        super(attention_sentiment_score_network, self).__init__()
        self.vocab_embedding=vocab_embedding
        self.dropout=dropout
        self.MIN_VALUE=MIN_VALUE
        self.rnn = nn.GRU(
            input_size=vocab_embedding.weight.shape[1],
            hidden_size=bilstm_embed_size,   #这个参数决定了隐层的维度，其实也决定了输出的维度，输出embed维度=hidden_size*(2 if bidirectional==True else 1)
            num_layers=bilstm_num_layers,   #这个参数是GRU里隐层的层数，默认为1。这个值的设置与bidirectional无关，请注意这一点。
            bidirectional=bidirectional,
            batch_first=True     #True则输入输出的数据格式为 (batch, seq, feature)
        )
        self.word_sentiemnt_transform = nn.Sequential(
            nn.Linear( bilstm_embed_size*2 , 1),
           # nn.Dropout(dropout)
        )

        self.sentence_vector=nn.Parameter( torch.Tensor(2*bilstm_embed_size,1) )

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.sentence_vector)

    # #我觉得softmax分配atten太集中，换个自己的函数
    # score = score.masked_fill(sen_pad_mask, 0)
    # word_attention = self.hxs_softmax(score)
    def hxs_softmax(self,score):
        score=F.relu(score)+1e-9
        score_sum=score.sum(axis=1,keepdim=True)
        attention=score / score_sum
        return attention

    def forward(self, sentence, sen_1_mask, sen_2_mask):
        ## word_embedding
        word_embedding=self.vocab_embedding(sentence)
        # word_embedding.shape:[batch_size,max_sen_len,300]

        ## Bilstm
        word_embedding=F.dropout(word_embedding, p=self.dropout, training=self.training)
        # # rnn在测试时输入单个样本会报错，因为需要输入为三维
        # if word_embedding.dim()<3:
        #     word_embedding=word_embedding.unsqueeze(0)
        word_context,_ =self.rnn(word_embedding)
        # word_context.shape:[batch_size,max_sen_len,300*2]

        ## word_attention
        score=word_context.matmul( self.sentence_vector  )
        score=score.squeeze(-1)
        sen_pad_mask=( sentence == 0 )
        score=score.masked_fill( sen_pad_mask, self.MIN_VALUE )
        word_attention= F.softmax(score,dim=1)
        # word_context.shape:[batch_size,max_sen_len]

        ## word_sentiment
        word_sentiment=self.word_sentiemnt_transform( word_context )
        word_sentiment=word_sentiment.squeeze(-1)
        word_sentiment=torch.sigmoid(word_sentiment)
        word_sentiment=( word_sentiment-0.5)*2 #值域变为[-1,1]
        # word_context.shape:[batch_size,max_sen_len]

        #我需要
        return word_attention,word_sentiment

class DSC_train_network(nn.Module):
    def __init__(self, dropout, vocab_embedding, bilstm_embed_size, bilstm_num_layers,
                 bidirectional,MIN_VALUE):
        super(DSC_train_network, self).__init__()
        self.attention_sentiment_score_model=attention_sentiment_score_network(dropout, vocab_embedding,
                                       bilstm_embed_size, bilstm_num_layers,bidirectional,MIN_VALUE)

    def forward(self, sentence, sen_1_mask, sen_2_mask):
        ## 计算word_attention和word_sentiment
        word_attention, word_sentiment=self.attention_sentiment_score_model(sentence,
                                                                                   sen_1_mask, sen_2_mask)

        ##计算整个句子的sentiment
        # 计算 word_sentence_sentiment
        word_sentence_sentiment=word_attention*word_sentiment
        #计算句子前半部分的sentiment
        sentence_1_sentiment=word_sentence_sentiment*sen_1_mask
        sentence_1_sentiment=sentence_1_sentiment.sum(axis=1)
        sentence_1_sentiment=sentence_1_sentiment.view( (sentence_1_sentiment.shape[0],1)  )
        # 计算句子后半部分的sentiment
        sentence_2_sentiment=word_sentence_sentiment*sen_2_mask
        sentence_2_sentiment=sentence_2_sentiment.sum(axis=1)
        sentence_2_sentiment=sentence_2_sentiment.view( (sentence_2_sentiment.shape[0],1)  )
        #拼接两部分sentiment
        sentence_sentiment=torch.cat( (sentence_1_sentiment, sentence_2_sentiment ), 1 )
        # sentence_sentiment.shape:[batch_size,2]

        #####增加个loss，要求大多数单词的word_sentiment都接近0
        sen_mask = (sentence != 0)
        word_sentiment_average=torch.abs( ( sen_mask*torch.abs( word_sentiment ) ).sum() / sen_mask.sum() -0.25 )

        # word_sentiment_average=torch.abs( ( torch.abs( word_sentiment ) ).mean()-0.25 )

        return sentence_sentiment,word_sentiment_average











