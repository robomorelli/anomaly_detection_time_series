import os

paths_to_exclude = ['data', 'dataloader', 'dataset', 'model_result', 'models', 'notebook', 'esa']
paths = []
root = os.getcwd()
root_parts = root.split('/')
root = [x if x not in paths_to_exclude else '' for x in root_parts]
root = '/'.join(root)
model_results = os.path.join(root, 'model_results/')
data_path = os.path.join(root, 'data/')


columns = ['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr',  'RW1_cmd_volt', 'RW2_cmd_volt',
       'RW3_cmd_volt', 'RW4_cmd_volt', 'RW1_therm',
       'RW2_therm', 'RW3_therm', 'RW4_therm', 'RW1_speed', 'RW2_speed',
       'RW3_speed', 'RW4_speed']

#['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr', 'RW1_therm',
#'RW2_therm', 'RW3_therm', 'RW4_therm', 'RW1_speed', 'RW2_speed',
#'RW3_speed', 'RW4_speed', 'RW1_cmd_volt', 'RW2_cmd_volt',
#'RW3_cmd_volt', 'RW4_cmd_volt', 'RW1_cmd_sign', 'RW2_cmd_sign',
#'RW3_cmd_sign', 'RW4_cmd_sign', 'AOC_mode', 'AOC_helper_1']