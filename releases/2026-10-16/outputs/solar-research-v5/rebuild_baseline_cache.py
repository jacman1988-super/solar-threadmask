"""Rebuild v3 reference IoU matrices from authorized local images and masks."""
from research import Dataset,ROOT,OLD,DATA
from experiment_v3 import Experiment

if __name__=='__main__':
    d=Dataset();e=Experiment(DATA,OLD/'results/v3',ROOT/'work/solar-v3-cache')
    for k,name in enumerate(d.protocol['split']['assessment'],1):
        e.record(name)
        print('Rebuilt baseline reference',k,flush=True)
