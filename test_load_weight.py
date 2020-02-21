import sys
import os
import cfs_coco_train as coco
from calculate import load_weight

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as mrcnn

DEFAULT_LOGS_DIR = coco.CocoConfig().DEFAULT_LOGS_DIR
SAVE_MODEL_DIR = os.path.join(DEFAULT_LOGS_DIR, 'exp_generated')

# ...................................................................................
folder_name = input('Which folder in logs saving the weights? Enter: ')
model_index = int(input('Enter model index (m100 then enter 100): '))
model_path = os.path.join(DEFAULT_LOGS_DIR, folder_name)
model_name = 'mask_rcnn_coco_'
load_model = (str(model_index).zfill(4)) + '.h5'

model_fullname = model_name + load_model
current_model_path = os.path.join(model_path, model_fullname)
print(current_model_path + '\n')
current_model = load_weight(current_model_path, coco.CocoConfig())

print('Load successfully.')
