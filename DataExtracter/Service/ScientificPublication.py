import requests
import os
from utils.Logger import logger
import shutil
import time
import random

def extract(metadata, video_dir):
    paper_df = metadata[(metadata.source == 'Paper')] 

    for idx, row in paper_df.iterrows():
        filename = row.id + '.' + row.filetype
        
        r = requests.get(row.url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            with open(os.path.join(video_dir, filename), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)       
            
        if (('241_' in row.id) | ('242_' in row.id) | ('243_' in row.id)): 
            delay = random.randint(10, 20)
        else:
            delay = random.randint(3, 5)
        time.sleep(delay)
    logger.info('Video files extraction from papers is done!')        
        