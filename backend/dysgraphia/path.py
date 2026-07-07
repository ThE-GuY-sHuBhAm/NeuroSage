from pathlib import Path

BASE = Path('.')

#? ML
DEVICE = 'cuda:0'
CHECKPOINTS = 'checkpoints'

#? IAM dataset
IAM = BASE / 'IAM'
XML = IAM / 'xml'
SETS = IAM / 'SETS'
DATA = IAM / 'DATA'

#? Dysgraphia dataset
DYSG = BASE / 'data'
CSVS = DYSG / 'csv'