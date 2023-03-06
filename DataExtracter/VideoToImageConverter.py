import pandas as pd
import cv2
import os
from utils.Logger import logger
import shutil

def label_to_dir(lab):
    if lab == "Cov":
        label = "covid"
    elif lab == "Pne" or lab == "pne":
        label = "pneumonia"
    elif lab == "Reg":
        label = "normal"
    elif lab == "Vir":
        label = "pneumonia"
    else:
        raise ValueError("Wrong label! " + lab)
    return label

def video_to_image(out_image_dir, label, video_path, file_name_params, FRAMERATE=1, all_frames=True):
    os.makedirs(out_image_dir, exist_ok=True)
    out_dir = os.path.join(out_image_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_image_dir, label)

    cap = cv2.VideoCapture(
        video_path
    )  
    frameRate = cap.get(5)  
    
    frame_markup = 1  if all_frames else int(frameRate / FRAMERATE)

    while cap.isOpened():
        frameId = cap.get(1) 
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % frame_markup == 0):
            filename = os.path.join(out_dir, file_name_params['file_id'] + "_" + file_name_params['video_probe'] + "_frame%d.jpg" % frameId)
            cv2.imwrite(filename, frame)
        
    cap.release()

def extract_images(video_path, cropped):
    if cropped:
        vid_prop_df = pd.read_csv('utils/video_cropping_metadata.csv', sep=',', encoding='latin1')
    else:
        metadata = pd.read_csv(os.environ['VIDEO_META_DATA'], sep=',', encoding='latin1')
        metadata = metadata[metadata.id !='22_butterfly_covid'] 

        vid_prop_df = pd.read_csv(os.environ['VIDEO_FILE_PROPERTIES'])
        vid_prop_df = vid_prop_df[vid_prop_df.filename !='22_butterfly_covid.mp4'] 

        vid_prop_df.filename = vid_prop_df.filename.astype(str)
        vid_prop_df.filename = vid_prop_df.filename.str.strip()

        metadata['filename2'] = metadata.id + '.' + metadata.filetype
        metadata.filename2 = metadata.filename2.astype(str)
        metadata.filename2 = metadata.filename2.str.strip()

        vid_prop_df = pd.merge(vid_prop_df, metadata[['filename2', 'source', 'probe', 'class']], left_on='filename', right_on='filename2', how='left').drop('filename2', axis=1)
        del metadata['filename2']

    vid_prop_df = vid_prop_df.rename(columns={'class': 'class_'})

    for idx, row in vid_prop_df.iterrows():
        if cropped:
            filename = row.filename.split('.')[0] + '_prc.avi'
            file_id = filename.split('.')[0]
        else:
            filename = row.filename
            file_id = row.filename.split('.')[0]
            
        vid_probe = str(row.probe).lower()
        class_ = str(row.class_).lower()

        video_to_image(os.environ['IMAGE_DIR'], class_, os.path.join(video_path, filename), {'video_probe': vid_probe, 'file_id': file_id})


def remove_outliers(cropped):
    image_prc_df = pd.read_csv('utils/mask_metadata.csv')
    count = 0
    image_prc_df = image_prc_df[image_prc_df.filename !='22_butterfly_covid.mp4'] # 22_butterfly_covid.mp4 was removed in March release of butterfly
    image_prc_df = image_prc_df.rename(columns={'class': 'class_'})
    for idx, row in image_prc_df[~pd.isna(image_prc_df.delete_frames_from_to)].iterrows():
        frames_to_delete = row.delete_frames_from_to.strip().split(',')
        if cropped:
            frame_name_main = row.mask_main_filename.split('.')[0].split('_frame')[0]
        else:
            frame_name_main = row.filename.split('.')[0] + "_" + row.probe.lower()

        
        for frames in frames_to_delete:
            from_frame = int(frames.split('-')[0])
            to_frame = int(frames.split('-')[1]) + 1
            
            for i in range(from_frame, to_frame):
                count += 1
                file_to_remove =  frame_name_main + '_frame' + str(i) + '.jpg'
                file_path = os.path.join('data/frames', row.class_.lower(), file_to_remove)
                if os.path.exists(file_path):
                    print(file_path, ' is removed ')
                    os.remove(file_path)
                else:
                    print(file_path, ' is missing ')

    logger.info("=== Files removed! ===")


def extract_images_from_pocus_data():
    POCUS_IMAGE_DIR = os.environ['POCUS_IMAGE_DIR']
    POCUS_VID_DIR = os.environ['POCUS_VID_DIR']
    modes = ['convex', 'linear']
    image_formats = ["png", "jpg", "peg", "JPG", "PNG"]
    out_image_dir = os.environ['FINAL_IMG_DIR']
    video_formats = ["peg", "gif", "mp4", "m4v", "avi", "mov"]
    req_classes = ["covid", "pneumonia", "normal"]
    img_format = '.jpg'
    for mode in modes:
        
        for fp in os.listdir(os.path.join(POCUS_IMAGE_DIR, mode)):
            if fp[-3:] in image_formats:
                label_dir = label_to_dir(fp[:3])
                os.makedirs(out_image_dir, exist_ok=True)
                os.makedirs(os.path.join(out_image_dir, label_dir), exist_ok=True)
                if label_dir in req_classes:

                    logger.info(os.path.join(POCUS_IMAGE_DIR, mode, fp))
                    logger.info(os.path.join(out_image_dir, label_dir, fp[:-4]+img_format))
                    shutil.copy(
                        os.path.join(POCUS_IMAGE_DIR, mode, fp),
                        os.path.join(out_image_dir, label_dir, fp[:-4]+img_format)
                    )

    for mode in modes:
        vid_files = os.listdir(os.path.join(POCUS_VID_DIR, mode))
        for i in range(len(vid_files)):

            if vid_files[i][-3:].lower() not in video_formats:
                    continue
            
            video_path = os.path.join(POCUS_VID_DIR, mode, vid_files[i])
            label = label_to_dir(vid_files[i][:3])
            if label not in req_classes:
                continue
            video_to_image(out_image_dir, label, video_path, {'file_id': vid_files[i][:-4], 'video_probe': mode})


