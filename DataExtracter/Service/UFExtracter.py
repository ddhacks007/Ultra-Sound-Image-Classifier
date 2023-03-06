  

import requests
import os
from utils.Logger import logger
import random
import time
import shutil

def extract(metadata, video_dir, source):
    if source != 'UF':
        paper_df = metadata[(metadata.source == 'Paper') & ((metadata['id'].str.contains('199', na=False)) | (metadata['id'].str.contains('200', na=False)))] 
    else:
        paper_df = metadata[metadata.source == 'UF']

    for idx, row in paper_df.iterrows():
        filename = row.id + '.' + row.filetype
        
        r = requests.get(row.url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            with open(os.path.join(video_dir, filename), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)       

        delay = random.randint(3, 5)
        time.sleep(delay)
    
    if source != 'UF':
        extract(metadata, video_dir, 'UF')
    else:
        return

    logger.info('UF extra video files downloaded! ===')      