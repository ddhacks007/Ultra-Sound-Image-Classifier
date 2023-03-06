import requests
import os
from utils.Logger import logger

def extract(metadata, video_dir):
    pocus_df = metadata[metadata.source == 'PocusAtlas']

    for idx, row in pocus_df.iterrows():
        filename = row.id + '.' + row.filetype
        vid = requests.get(row.url).content
        with open(os.path.join('data/video/', filename), 'wb') as handler:
            handler.write(vid)
    logger.info(' video files extraction done!')        