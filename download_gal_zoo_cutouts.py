"""
This code is taken from the Galaxy Classification Project by Christopher Akroyd and Artem Barinov
which was created as part of a Coursework Project.

License is included in LICENSE.md
"""

import pandas as pd
from tqdm import tqdm
import pathlib
from urllib import request
from multiprocessing.pool import ThreadPool
import threading

# Path to the data.
gal_data_path = './data/galaxy_data/'
path_to_gz2_csv = './data/gz2_hart16.csv'
# The columns we actually want to read from the galaxy zoo CSV.
wanted_cols = ['dr7objid', 'ra', 'dec', 'gz2_class']
# Sets of galaxy classes
galaxy_classes = {'Ec', 'Ei', 'Er', 'Sa', 'Sb', 'Sc', 'SBa', 'SBb', 'SBc'}


def img_download(object_id, obj_class, ra, dec):
    """
    Function that when given an right ascension and declination values downloads an image from a SDSS mirror
    for Data Release 8. The image is then saved within a folder for its class with its given object id.
    :param object_id: The Galaxy Zoo Object ID for this image.
    :param obj_class: The Galaxy Class.
    :param ra: Right Ascension value from the Galaxy Zoo .csv.
    :param dec: The Declination value from the Galaxy Zoo .csv.
    :return:
    """
    url = 'http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra={}&dec={}&scale=0.2&width=240&height=240&opt='\
        .format(ra, dec)
    outfile = gal_data_path + '{}/{}.jpg'.format(obj_class, object_id)
    image = request.URLopener()
    image.retrieve(url, outfile)


def create_dirs():
    '''
    Create directories to store images of each class in.
    '''
    for gal_class in galaxy_classes:
        pathlib.Path(gal_data_path + gal_class).mkdir(parents=True, exist_ok=True)


def download_one_class(gal_zoo, gal_class):
    """
    Filters a data frame of images to only contain the specified galaxy class.
    :param gal_zoo: Full Dataframe.
    :param gal_class: The Galaxy Class to download.
    :return: A data frame with values for only one class.
    """
    return gal_zoo[gal_zoo.gz2_class.str.contains(gal_class, regex=True, na=False)]


def download_images(one_class=None, position=0, lock=threading.Lock()):
    """
    Thread safe downloader for Galaxy Zoo Images with a TQDM progress bar.
    :param one_class: Whether to download only items of one class.
    :param position: Position of the TQDM progress bar (Default 0)
    :param lock: Lock for updating the progress bar.
    """
    gal_zoo = pd.read_csv(path_to_gz2_csv, usecols=wanted_cols)
    # We don't care about stars so remove them.
    gal_zoo = gal_zoo[gal_zoo.gz2_class != 'A']
    # We want to remove any images which contain odd or unusual features
    # e.g. ‘(r)’=ring, ‘(l)’=lens/arc, ‘(d)’=disturbed, ‘(i)’=irregular,
    # ‘(o)’=other, ‘(m)’=merger, ‘(u)’=dust lane.
    # Items with these features are represented as SBc(t) etc.
    gal_zoo = gal_zoo[~gal_zoo.gz2_class.str.contains('\([a-z]\)', regex=True, na=False)]
    # Remove edge on disks
    gal_zoo = gal_zoo[~gal_zoo.gz2_class.str.contains('(Se)[rnb]', regex=True, na=False)]
    # Remove classes with few data items
    gal_zoo = gal_zoo[~gal_zoo.gz2_class.str.contains('(Sd)|(SBd)', regex=True, na=False)]

    if one_class:
        gal_zoo = download_one_class(gal_zoo, one_class)

    with lock:
        progress = tqdm(
            total=len(gal_zoo),
            position=position,
        )

    for dr7objid, ra, dec, gz2_class in gal_zoo.itertuples(index=False):
        hubble_sequence_class = ''
        if gz2_class[:2] in galaxy_classes:
            hubble_sequence_class = gz2_class[:2]
        elif gz2_class[:3] in galaxy_classes:
            hubble_sequence_class = gz2_class[:3]
        else:
            pass

        img_download(dr7objid, hubble_sequence_class, ra, dec)
        # Update the progress bar
        with lock:
            progress.update(1)
    with lock:
        progress.close()


if __name__ == '__main__':
    # Create directories for each class.
    create_dirs()
    # Read in the Galaxy Zoo CSV.
    gal_zoo = pd.read_csv(path_to_gz2_csv, usecols=wanted_cols)
    # Init the multi-threaded environment and assign 4 threads per each galaxy class. Each class is then downloaded
    # individually within their own thread.
    pool = ThreadPool(len(galaxy_classes) * 4)
    lock = threading.Lock()
    for i, gal_class in enumerate(galaxy_classes):
        pool.apply_async(download_images, args=(gal_class, i, lock))
    # Cleanup.
    pool.close()
    pool.join()
