import json
import os

from odometry.runner import runAnomalyTraining
from odometry.utils import default_logger
from odometry.utils.io import read_json


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

train_config = read_json(os.path.join(CURRENT_DIR, 'train_config.json'))
runAnomalyTraining(conf=train_config)