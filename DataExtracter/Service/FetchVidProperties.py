
import os
import cv2
from PIL import Image
from utils.Logger import logger

def extract_vid_properties(video_path): 

    vid_files = [file for file in os.listdir(video_path) if not file == '.DS_Store']
    os.remove(os.environ['VIDEO_FILE_PROPERTIES']   )
    with open(os.environ['VIDEO_FILE_PROPERTIES'], 'w') as f:
        f.write('filename,framerate,width,height,frame_count,duration_secs\n')
        
        for vid in vid_files:
            vid_filename = os.path.join(video_path, str(vid))
            file_type = vid.split('.')[-1]
            cv2video = cv2.VideoCapture(vid_filename)
            height = cv2video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width  = cv2video.get(cv2.CAP_PROP_FRAME_WIDTH) 
            frame_rate = round(cv2video.get(cv2.CAP_PROP_FPS), 2)
            
            if file_type == 'mp4':
                frame_count = cv2video.get(cv2.CAP_PROP_FRAME_COUNT) 
                duration = round((frame_count / frame_rate), 2)
            elif file_type == 'gif':
                frame_count = round(Image.open(vid_filename).n_frames) #round((duration * frame_rate ), 0)
                duration = round((frame_count / frame_rate), 2)

            line_to_write = str(vid) + ',' + str(frame_rate) + ',' + str(width) + ',' + str(height) + ',' + str(frame_count) + ',' + str(duration) + '\n'
            f.write(line_to_write)
            
    logger.info("Extracted video folder ")

