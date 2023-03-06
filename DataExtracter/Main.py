
import os
import pandas as pd
from Service import ButterflyExtracter, GrepMedEctracter, LiftlExtracter, PocusAltasExtracter, RadiopiaExtracter, CoreUltraSoundExtracter, UFExtracter, ScientificPublication, ClaurisService, FetchVidProperties
from VideoToImageConverter import extract_images, extract_images_from_pocus_data, remove_outliers 
import shutil
import time
from utils.Logger import logger
from Service.MediaCropper import crop_video, remove_markers

if __name__ == '__main__':
    start = time.time()
    DATA_VID_DIR = os.environ['DATA_VID_DIR']
    BUTTERFLY_VID_DIR = os.environ['BUTTERFLY_VID_DIR']
    CLAURIS_VID_DIR = os.environ['CLAURIS_VID_DIR']
    CROPPED_VID_DIR = os.environ['CROPPED_VID_PATH']
    IMG_DIR = os.environ['IMAGE_DIR']
    FINAL_IMG_DIR = os.environ['FINAL_IMG_DIR']
    metadata = pd.read_csv(os.environ['VIDEO_META_DATA'], sep=',', encoding='latin1')

    if(os.path.exists(DATA_VID_DIR)):
        shutil.rmtree(DATA_VID_DIR)
    
    if not os.path.exists(FINAL_IMG_DIR):
        os.makedirs(FINAL_IMG_DIR, exist_ok=True)

    os.makedirs(DATA_VID_DIR, exist_ok=True)
    os.makedirs(CROPPED_VID_DIR, exist_ok=True)

    ButterflyExtracter.export_and_rename_bfly_wrt_meta(BUTTERFLY_VID_DIR, DATA_VID_DIR, metadata)
    GrepMedEctracter.extract(metadata, DATA_VID_DIR)
    LiftlExtracter.extract(metadata, DATA_VID_DIR)
    PocusAltasExtracter.extract(metadata, DATA_VID_DIR)
    RadiopiaExtracter.extract(metadata, DATA_VID_DIR)
    CoreUltraSoundExtracter.extract(metadata, DATA_VID_DIR)
    UFExtracter.extract(metadata, DATA_VID_DIR,'!UF')
    ScientificPublication.extract(metadata, DATA_VID_DIR)
    ClaurisService.extract(metadata, DATA_VID_DIR, CLAURIS_VID_DIR)
    logger.info(f"cropping videos ....")
    crop_video(DATA_VID_DIR, CROPPED_VID_DIR)
    logger.info(f"cropped successfully!")
    logger.info("extracting video properties")
    FetchVidProperties.extract_vid_properties(DATA_VID_DIR)
    if os.path.exists(IMG_DIR):
        shutil.rmtree(IMG_DIR)
    extract_images(CROPPED_VID_DIR, True)
    remove_outliers(True)
    logger.info(f"removing the markers")
    for folder in os.listdir(IMG_DIR):
        directory_path = os.path.join(IMG_DIR, folder)
        if folder != 'other':
            outputdir_path = os.path.join(FINAL_IMG_DIR, folder)
            os.makedirs(outputdir_path, exist_ok=True)
            remove_markers(directory_path, outputdir_path)
    
    # extract_images_from_pocus_data()    

    logger.info(f"successfully removed the markers")

    logger.info(f"Total time taken for processing the data is {time.time() - start} seconds")


