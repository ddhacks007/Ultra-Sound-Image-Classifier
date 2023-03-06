import os
from utils.Logger import logger
from vimeo_downloader import Vimeo
import requests

def extract(metadata, video_dir):
    core_df = metadata[metadata.source == 'CoreUltrasound']

    for idx, row in core_df.iterrows():
        filename = row.id + '.' + row.filetype
        
        if 'vimeo' in row.url:
            v = Vimeo(row.url)
            stream = v.streams 
            highest_quality_available = stream[-1]
            highest_quality_available.download(download_directory = video_dir, filename = filename.split('.')[0])
        else:
            vid = requests.get(row.url).content
            with open(os.path.join(video_dir, filename), 'wb') as handler:
                handler.write(vid)
    logger.info('CoreUltrasound video files extraction done! ===')        