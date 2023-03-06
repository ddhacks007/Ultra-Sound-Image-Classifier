import os
from utils.Logger import logger

def export_and_rename_bfly_wrt_meta(BUTTERFLY_VID_DIR, DATA_VID_DIR, metadata):
    def rename(metadata, DATA_VID_DIR):
        for root, dirs, files in os.walk(DATA_VID_DIR):  
            for file in files:
                if file.endswith(".png") or file == '.DS_Store':
                    continue
                path_file = os.path.join(root,file)
                try:
                    file_id = metadata[metadata.filename == file].id.values[0] + '.mp4'
                    os.rename(path_file, os.path.join(root,file_id))
                    
                except Exception as e:
                    logger.error(f"{e} {file} occurred in rename of butterfly extracter")

    for root, dirs, files in os.walk(BUTTERFLY_VID_DIR):  
        for file in files:
            if file.endswith(".png"):
                continue
            path_file = os.path.join(root,file)
            os.system(f'cp "{path_file}" {DATA_VID_DIR}') 
    rename(metadata, DATA_VID_DIR)
    logger.info(f"butterfly extraction successful!")
    
