import os
import yaml
from ASC.train.train import train

# 强迫pytorch里边的gpu编号与任务管理器里边的gpu编号一致，方便查看
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



config = yaml.safe_load(open('config.yml'))
config=config['ASC']
# data_source='restaurant'
data_source = 'laptop'
os.environ["CUDA_VISIBLE_DEVICES"] = str( config['CUDA_VISIBLE_DEVICES'] )




if __name__ == '__main__':
    train(config, data_source)