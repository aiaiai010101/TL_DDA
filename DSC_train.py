import yaml
from DSC.train.train import train

config = yaml.safe_load(open('config.yml'))
config=config['DSC']
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])
#os.environ['CUDA_ENABLE_DEVICES'] = '0'
#torch.backends.cudnn.enabled = False #老是报错cudnn，干脆禁了，果然不报错了，不过训练速度慢一倍
if __name__ == '__main__':
    train(config)