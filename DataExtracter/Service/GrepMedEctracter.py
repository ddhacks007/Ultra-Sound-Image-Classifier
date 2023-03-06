
import requests
import os
from utils.Logger import logger

def extract(metadata, video_dir):
    grepmed_df = metadata[metadata.source == 'GrepMed']

    for idx, row in grepmed_df.iterrows():
        filename = row.id + '.' + row.filetype
        vid = requests.get(row.url).content
        with open(os.path.join(video_dir, filename), 'wb') as handler:
            handler.write(vid)
    logger.info(' GrepMed video files extraction done!')      

  