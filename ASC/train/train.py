# 使用 os.getcwd() 获取当前路径
import os
from ASC.train.utils  import *
from ASC.train.network import *
from torch import optim
import numpy as np
import time
import random


def train(config,data_source):

    # 设置最多使用几个CPU线程
    torch.set_num_threads(1)

    # 保存训练日志
    train_log_path = config['train_log_dir'] + '/' + data_source + '_train_log.txt'
    all_train_log_path = config['train_log_dir'] + '/' + data_source + '_all_train_log.txt'
    if os.path.exists(train_log_path):
        os.remove(train_log_path)
    if os.path.exists(all_train_log_path):
        os.remove(all_train_log_path)

    for experiment_time in range(10000):
        # 指定随机数种子
        # seed=random.randint(0,2**32-1)
        seed=1428108170
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
    #制作数据加载器
        train_loader,test_loader=make_data_loader(config,data_source)
    #制作模型
        device=config['device']
        model=make_model(config,data_source)
        model = model.to(device)
    #代价函数   ###########
        # criterion = nn.MSELoss(reduction='mean')
        criterion=nn.CrossEntropyLoss()
    #制作优化器
        optimizer=optim.Adam(model.parameters(), lr=config['net_parameter']['learning_rate'],
                              weight_decay=config['net_parameter']['weight_decay'])

    #制作各种需要显示存储的loss
        min_last_10_average_loss=10000
        #设置一个list，存储最近10个step的loss，方便后边显示，因为一个epoch很耗时间，可以看看最近n个step的平均loss
        last_10_loss=[-1]*10
        max_test_accuracy=0
        max_test_accuracy_epoch=0

        for epoch in range(config['net_parameter']['num_epoches']):

            start = time.time()
            #计算每个epoch的average_loss
            epoch_sum_loss=0
            sample_sum_num=0

            for i, data in enumerate(train_loader):
                # data是一个list，有9个元素，分别对应一次训练使用的['sentence', 'label', 'sen_len',
                # 'target_word', 'target_len', 'position', 'attention_1', 'pos', 'syntax_position']
                # data[0].shape:[batch_size,max_sentence_len]
                # data[1].shape:[batch_size]
                # data[2].shape:[batch_size,max_sentence_len]
                # data[3].shape:[batch_size,max_target_len]
                # data[4].shape:[batch_size]
                # data[5].shape:[batch_size,max_sentence_len]
                # data[6].shape:[batch_size,max_sentence_len]

                model.train() #开启dropout
                # 取出本次batch的训练数据
                train_sentence, train_label, train_sen_len, train_target_word, \
                train_target_len, train_position, train_attention_1, train_pos, train_syntax_position=data

                train_sentence, train_label, train_sen_len, train_target_word, \
                train_target_len, train_position, train_attention_1, train_pos, train_syntax_position = \
                    train_sentence.to(device), train_label.to(device), train_sen_len.to(device), train_target_word.to(device), \
                    train_target_len.to(device), train_position.to(device), train_attention_1.to(device), train_pos.to(device),\
                    train_syntax_position.to(device)

                # 所有参数梯度零值化
                optimizer.zero_grad()
                # 送入model，得到返回值
                logit=model( train_sentence, train_sen_len, train_target_word, train_target_len,
                             train_position, train_attention_1, train_pos, train_syntax_position)
                train_label_scalar=torch.max(train_label, 1)[1]
                # train_label_scalar=(train_label_scalar-1)*(-1.)

                loss=criterion(logit, train_label_scalar)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                # 计算需要显示的loss
                epoch_sum_loss=epoch_sum_loss+loss*train_sentence.shape[0]
                sample_sum_num=sample_sum_num+train_sentence.shape[0]

                #保存最近10个step的loss
                last_10_loss.append( float(loss) )
                del last_10_loss[0]

                last_10_loss_mask=np.array(last_10_loss)>-0.5
                last_10_average_loss=  ( np.array(last_10_loss_mask)*last_10_loss ).sum()/last_10_loss_mask.sum()

                if last_10_average_loss<min_last_10_average_loss:
                    min_last_10_average_loss=last_10_average_loss

                ## 计算准确率
                test_accuracy,test_loss=eval(model, test_loader, criterion, device)
                if test_accuracy>max_test_accuracy:
                    max_test_accuracy=test_accuracy
                    max_test_accuracy_epoch=epoch
                #显示
                if i % 10 == 0 and i >= 0:
                    train_accuracy=get_accuracy(logit, train_label_scalar)
                    display='[seed %2d] [experiment_time %2d] [epoch %2d] [step %3d] train_loss: %.4f train_accuracy: %.4f last_10_average_loss: %.4f min_last_10_average_loss: %.4f ' \
                            '************** test_loss: %.4f test_accuracy: %.4f max_test_accuracy: %.4f max_test_accuracy_epoch: %2d'\
                            % (seed, experiment_time, epoch, i, loss, train_accuracy, last_10_average_loss,
                                             min_last_10_average_loss, test_loss, test_accuracy, max_test_accuracy, max_test_accuracy_epoch)
                    print( display  )
                    #保存训练日志
                    train_log = open(train_log_path, 'a')
                    train_log.writelines(  display+'\n' )
                    train_log.close()

            #计算并显示每个epoch的average_loss
            epoch_average_loss= epoch_sum_loss/sample_sum_num
            print('[epoch %2d] epoch_average_loss: %.4f' % (epoch,  epoch_average_loss))
            end = time.time()
            print('time: %.4fs' % (end - start))
        # 保存训练日志
        all_train_log = open(all_train_log_path, 'a')
        all_train_log.writelines('[seed %2d] [experiment_time %2d] max_test_accuracy: %.4f max_test_accuracy_epoch: %2d' % (seed, experiment_time, max_test_accuracy, max_test_accuracy_epoch) + '\n')
        all_train_log.close()



