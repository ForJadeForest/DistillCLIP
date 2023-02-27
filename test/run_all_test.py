import os

python_script_list = ['flickr8k_ex.py', 'composite.py', 'pascal-50s_ex.py', 'coco_ex.py', 'FOIL_allucination.py']

for script in python_script_list:
    os.system(f'~/.conda/envs/pyz/bin/python {script}')