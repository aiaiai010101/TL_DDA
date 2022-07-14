# 使用 os.getcwd() 获取当前路径
from DSC.train.utils  import *
from DSC.train.network import *
from torch import optim
import numpy as np
import time


def train(config):
#制作数据加载器
    train_loader=make_data_loader(config)
#制作模型
    device=config['device']
    model=make_model(config)
    model = model.to(device)
#代价函数
    criterion = nn.MSELoss(reduction='mean')
#制作优化器
    optimizer=optim.Adam(model.parameters(), lr=config['net_parameter']['learning_rate'],
                         weight_decay=config['net_parameter']['weight_decay'])

#制作各种需要显示存储的loss
    min_last_100_average_loss=10000
    #设置一个list，存储最近n个step的loss，方便后边显示，因为一个epoch很耗时间，可以看看最近n个step的平均loss
    last_100_loss=[-1]*100
    epoch_average_loss = -1  # -1表示还没经历1个epoch
    last_save_100_average_loss=10000

    for epoch in range(config['net_parameter']['num_epoches']):

        start = time.time()
        #计算每个epoch的average_loss
        epoch_sum_loss=0
        sample_sum_num=0

        for i, data in enumerate(train_loader):
            # data是一个list，有4个元素，分别对应一次训练使用的['sentence', 'label', 'sen_1_mask', 'sen_2_mask']
            # data[0].shape:[batch_size,max_sentence_len]
            # data[1].shape:[batch_size,2]
            # data[2].shape:[batch_size,max_sentence_len]
            # data[3].shape:[batch_size,max_sentence_len]
            model.train() #开启dropout
            # 取出本次batch的训练数据
            sentence, label, sen_1_mask, sen_2_mask=data
            sentence, label, sen_1_mask, sen_2_mask = sentence.to(device), label.to(device), \
                                                       sen_1_mask.to(device), sen_2_mask.to(device)

            # 所有参数梯度零值化
            optimizer.zero_grad()
            # 送入model，得到返回值
            logit=model( sentence, sen_1_mask, sen_2_mask)
            ## 计算loss
            weight=1  #两个loss之间的权重系数
            loss_1=criterion(logit[0], label)
            loss_2=logit[1]
            loss=loss_1+weight*loss_2
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 计算需要显示的loss
            epoch_sum_loss=epoch_sum_loss+loss*sentence.shape[0]
            sample_sum_num=sample_sum_num+sentence.shape[0]

            #保存最近100个step的loss
            last_100_loss.append( float(loss) )
            del last_100_loss[0]

            # 保存模型
            last_100_loss_mask=np.array(last_100_loss)>-0.5
            last_100_average_loss=  ( np.array(last_100_loss_mask)*last_100_loss ).sum()/last_100_loss_mask.sum()

            if  last_save_100_average_loss-last_100_average_loss>0.01  :
                last_save_100_average_loss=last_100_average_loss
                #'DSC_checkpoint_model.pth'
                lr=config['net_parameter']['learning_rate']
                batch_size=config['net_parameter']['batch_size']
                checkpoint_model_path=config['checkpoint_model_dir']+'DSC_checkpoint_model_' \
                                             '[epoch_%3d]_' \
                                             '[last_100_average_loss_%.4f]_' \
                                             '[min_last_100_average_loss_%.4f]_' \
                                             '[epoch_average_loss_%.4f]_' \
                                             '[batch_size_%3d]_[lr_%.8f].pth'%\
                                      (epoch, last_100_average_loss,min_last_100_average_loss,epoch_average_loss,batch_size,
                                       lr)

                torch.save(model.state_dict(), checkpoint_model_path)
            if last_100_average_loss<min_last_100_average_loss:
                min_last_100_average_loss=last_100_average_loss
            #显示
            if i % 10 == 0 and i >= 0:
                print('[epoch %2d] [step %3d] train_loss: %.4f loss_1: %.4f loss_2: %.4f last_100_average_loss: %.4f min_last_100_average_loss: %.4f'
                                      % (epoch, i, loss, loss_1, loss_2, last_100_average_loss, min_last_100_average_loss) )
        #计算并显示每个epoch的average_loss
        epoch_average_loss= epoch_sum_loss/sample_sum_num
        print('[epoch %2d] epoch_average_loss: %.4f' % (epoch,  epoch_average_loss))
        #保存每一个epoch的模型，方便断点训练
        epoch_checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            }
        epoch_checkpoint_model_path = config['checkpoint_model_dir'] + 'DSC_epoch_checkpoint_model_' \
                                                           '[epoch_%3d]_' \
                                                           '[epoch_average_loss_%.4f]_' \
                                                           '[batch_size_%3d]_[lr_%8f].pth' % \
                             (epoch,  epoch_average_loss, batch_size,lr)

        torch.save(epoch_checkpoint, epoch_checkpoint_model_path)

        end = time.time()
        print('time: %.4fs' % (end - start))
