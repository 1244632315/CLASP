import os
import yaml
from pprint import pprint

from detect import DBT
from tool import get_fileList


def process_one_seq():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dbt = DBT(config)
    dbt.main()


def process_CLASP_seqs():
    dirpath = 'E:/CLASP/20220929'
    print(f'\n\n\nCurrent dirpath: {dirpath}')
    paths = get_fileList(dirpath, postfix='fits')
    tasks = {}
    for path in paths:
        name = os.path.basename(path)
        keys = name.split('-')
        sky, cam = keys[:2]
        key = sky + '-' + cam
        if key not in tasks.keys():
            tasks[key] = []
        tasks[key].append(path)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    for task, paths in tasks.items():
        print(f'\nStart to process task: {task}')
        config['task'] = task
        config['path'] = paths
        detector = DBT(config)
        detector.main()




if __name__ == '__main__':
    process_CLASP_seqs()