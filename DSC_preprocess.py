import yaml
from DSC.data_process.data_process import data_process

config = yaml.safe_load(open('config.yml'))
config=config['DSC']

data_process(config)