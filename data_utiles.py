#%%
import numpy as np
import pandas as pd
import scipy as sp

# For automatic dataset downloading
import urllib.request, urllib.error
from zipfile import ZipFile
from io import StringIO, BytesIO 
import shutil
import os.path


def download_dataset(dataset, files, data_dir):
    """
    Downloads dataset if files are not present.
    """
    if not np.all([os.path.isfile('./' + data_dir + f) for f in files]):
        url =  "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urllib.request.urlopen(url=url)

        print('Downloading {} dataset from {}'.format(dataset, url))
        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'data/' + dataset.replace('_', '-')
            
        elif dataset == 'ml_10m':
            target_dir = 'data/' + 'ml-10M'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('data/')

        source = [target_dir + '/' + s for s in os.listdir(target_dir)]
        destination = data_dir+'/'
        for f in source:
            shutil.copy(f, destination)

        shutil.rmtree(target_dir)

def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data: np.int32 arrays

    Returns
    -------
    mapped_data: np.int32 arrays
    n: length of mapped_data
    """
    #only uniq value
    uniq = set(data)

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n





#%%
