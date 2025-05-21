import os
import yaml

from detect import DBT
from tool import get_fileList

def process_CLASP_seqs(dir_path):
    # classify the task data
    paths = get_fileList(dir_path)
    tasks = {}
    for path in paths:
        name = os.path.basename(path)
        keys = name.split('-')
        sky, cam = keys[:2]
        if sky not in tasks.keys():
            tasks[sky] = []
        tasks[sky].append(path)
    
    # process each task independently
    for task, paths in tasks.items():
        print(f'\nStart to process task: {task}')
        detector = DBT(paths, task=task, postfix='fits', scale=0.2, method='SEP')
        detector.main()


def proecess_one_seq():
    task='SKYMAPPER0033'
    imgs = [x for x in get_fileList('./imgs/20221025') if task in x]
    detector = DBT(imgs, task=task, scale=0.2)
    detector.main()



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dbt = DBT(config)
    dbt.main()