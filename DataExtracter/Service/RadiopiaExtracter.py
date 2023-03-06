import requests
import os
from utils.Logger import logger

def extract(metadata, video_dir):
    radio_df = metadata[metadata.source == 'Radiopaedia']

    for idx, row in radio_df.iterrows():
        filename = row.id + '.' + row.filetype
        vid = requests.get(row.url).content
        with open(os.path.join(video_dir, filename), 'wb') as handler:
            handler.write(vid)
    logger.info('Radiopaedia video files extraction done! ===')        