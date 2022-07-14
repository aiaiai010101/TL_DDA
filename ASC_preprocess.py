import yaml
from ASC.data_process.data_process import data_process

config = yaml.safe_load(open('config.yml'))
# config=config['ASC']['laptop']
config=config['ASC']['restaurant']

data_process(config)