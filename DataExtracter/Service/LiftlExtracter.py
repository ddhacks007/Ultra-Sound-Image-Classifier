import requests
import os
from utils.Logger import logger

def extract(metadata, video_dir):
    litfl_df = metadata[metadata.source == 'Litfl']

    for idx, row in litfl_df.iterrows():
        filename = row.id + '.' + row.filetype
        vid = requests.get(row.url).content
        with open(os.path.join(video_dir, filename), 'wb') as handler:
            handler.write(vid)
    logger.info('LITFL video files extraction done!')  

  