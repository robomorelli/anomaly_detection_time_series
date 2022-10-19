import os

paths_to_exclude = ['data', 'dataloader', 'dataset', 'model_result', 'models', 'notebook']
paths = []
root = os.getcwd()
root_parts = root.split('/')
root = [x if x not in paths_to_exclude else '' for x in root_parts]
root = '/'.join(root)
model_results = os.path.join(root, 'model_results/')
data_path = os.path.join(root, 'data/')

columns = ['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr']
