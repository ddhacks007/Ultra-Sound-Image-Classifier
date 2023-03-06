import pandas as pd
import cv2
import os
import numpy as np
from utils.Logger import logger
import shutil
import re

def crop_video(VIDEO_PATH_ORG, VIDEO_CROPPED_OUT):
    
    vid_crp_metadata = pd.read_csv('utils/video_cropping_metadata.csv', sep=',', encoding='latin1')

    for idx, row in vid_crp_metadata.iterrows():
        vid_arr = []  
        
        filename = row.filename
        file_label = filename.split('_')[-1].split('.')[0] 
        
        if filename == '22_butterfly_covid.mp4':
            continue
        
        cap = cv2.VideoCapture(os.path.join(VIDEO_PATH_ORG, filename))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        dim = (width, height) 
        
        if pd.isna(row.x1_w_y1_h): 
            DEL_UPPER = int(row.del_upper) 
            WIDTH_RATE = float(row.width_rate) 
            
            width_border = int(width * WIDTH_RATE)
            width_box = int(width - (2 * width_border)) 
            if width_box + DEL_UPPER > height:
                width_box = int(height - DEL_UPPER)
                width_border = int( (width / 2) - (width_box / 2))

            while(True):
                ret, frame = cap.read()

                if not ret:
                    break

                frame = frame[DEL_UPPER:width_box + DEL_UPPER, width_border:width_box + width_border]

                frame = np.asarray(frame).astype(np.uint8)
                vid_arr.append(frame)

        else: 
            X1 = int(row.x1_w_y1_h.split(',')[0].replace('(', ''))
            W = int(row.x1_w_y1_h.split(',')[1].strip())
            Y1 = int(row.x1_w_y1_h.split(',')[2].strip())
            H = int(row.x1_w_y1_h.split(',')[3].replace(')', '').strip())

            while(True):
                ret, frame = cap.read()

                if not ret:
                    break

                frame = frame[Y1:Y1 + H, X1:X1 + W]

                frame = np.asarray(frame).astype(np.uint8)
                vid_arr.append(frame)

        vid_arr = np.asarray(vid_arr)
        prc_dim = vid_arr.shape[1:3] # dimension of the cropped file
        if len(prc_dim) >= 2:
            prc_dim = (prc_dim[1], prc_dim[0])

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(VIDEO_CROPPED_OUT, filename.split('.')[0] + '_prc.avi'), fourcc, 20.0, tuple(prc_dim))

            for frame in vid_arr:
                out.write(frame.astype("uint8"))

            vid_crp_metadata.iloc[idx, vid_crp_metadata.columns.get_loc('crp_width')] = prc_dim[1]
            vid_crp_metadata.iloc[idx, vid_crp_metadata.columns.get_loc('crp_height')] = prc_dim[0]

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            vid_crp_metadata.to_csv('utils/video_cropping_metadata.csv', index=None)

        logger.info(f' cropping done... for {filename.split(".")[0]}_prc.avi')


def zero_pad_array(arr, pad=5):
    if len(arr.shape) == 3:
        padded_arr = np.zeros((arr.shape[0]+2*pad, arr.shape[1]+2*pad, arr.shape[2]), dtype=np.uint8)
        padded_arr[pad:pad + arr.shape[0], pad:pad + arr.shape[1], :] = arr
    else:
        padded_arr = np.zeros((arr.shape[0]+2*pad, arr.shape[1]+2*pad), dtype=np.uint8)
        padded_arr[pad:pad + arr.shape[0], pad:pad + arr.shape[1]] = arr
    return padded_arr
        

def frame_inpainting(frame_dict, mask, default_mask=0, kernel_size=(5,5), method='telea', pad=5):
    kernel = np.ones(kernel_size, np.uint8)
    if type(mask) is not dict:
        mask = {default_mask: mask}
    masks_processed = {key:cv2.dilate(zero_pad_array(m, pad=pad), kernel, iterations=1) for key, m in mask.items()}
    
    method_dict = {'ns':cv2.INPAINT_NS, 'telea':cv2.INPAINT_TELEA}
    
    frames_inpainted = {}
    for key, frame in frame_dict.items():
        if key in masks_processed:
            frames_inpainted[key] = cv2.inpaint(zero_pad_array(frame, pad=pad), masks_processed[key], 3, method_dict[method])[pad:-pad, pad:-pad, :]
        else: 
            frames_inpainted[key] = cv2.inpaint(zero_pad_array(frame, pad=pad), masks_processed[default_mask], 3, method_dict[method])[pad:-pad, pad:-pad, :]

    return frames_inpainted

def remove_markers(IMAGE_CROPPED_OUT, CLEAN_IMAGE_OUT):
    IMAGE_MASK_OUT = os.environ['IMAGE_MASK_OUT']
    image_prc_df = pd.read_csv('utils/mask_metadata.csv')

    image_prc_df = image_prc_df[image_prc_df.filename !='22_butterfly_covid.mp4'] # 22_butterfly_covid.mp4 was removed in March release of butterfly


    for idx, row in image_prc_df.iterrows():     
        if row.probe == 'Convex':
            filename_main = row.filename.split('.')[0] + '_prc_convex'
        elif row.probe == 'Linear':
            filename_main = row.filename.split('.')[0] + '_prc_linear'
            
        if row.tight_inpainting == 'yes':
            inpainting_kernel_size = (1,1)
        else:
            inpainting_kernel_size = (5,5)

        if row.need_mask_after_crop == 'no':
            frames = {}
            
            for file in os.listdir():
                if file.startswith(filename_main):
                    new_filename = file.replace('frame', 'clean_frame')
                    shutil.copy(IMAGE_CROPPED_OUT + file, CLEAN_IMAGE_OUT + new_filename)

                    img = cv2.imread(os.path.join(CLEAN_IMAGE_OUT, new_filename))
                    frame_num = int(re.search(r'\d+$', file[:-4]).group())
                    frames[frame_num] = img
        else: 
            frames = {}
            for f in os.listdir(IMAGE_CROPPED_OUT):
                if f.startswith(filename_main):
                    img = cv2.imread(os.path.join(IMAGE_CROPPED_OUT, f))
                    frame_num = int(re.search(r'\d+$', f[:-4]).group())
                    frames[frame_num] = img

            if row.need_multiple_masks == 'no':
                mask = cv2.imread(os.path.join(IMAGE_MASK_OUT, filename_main + '_frame0_mask.jpg'))[:,:,0]

                frames_inpainted = frame_inpainting(frames, mask, kernel_size=inpainting_kernel_size)
            else:
                masks = {}
            
                for f in os.listdir(IMAGE_MASK_OUT):
                    if f.startswith(filename_main):
                        img = cv2.imread(os.path.join(IMAGE_MASK_OUT, f))
                        frame_num = int(re.search(r'\d+$', f[:-9]).group())
                        masks[frame_num] = img[:,:,0]

                frames_inpainted = frame_inpainting(frames, masks, default_mask=0, kernel_size=inpainting_kernel_size)

            for key, value in frames_inpainted.items():
                logger.info(os.path.join(CLEAN_IMAGE_OUT, filename_main + "_clean_frame" + str(key) + ".jpg"))
                cv2.imwrite(os.path.join(CLEAN_IMAGE_OUT, filename_main + "_clean_frame" + str(key) + ".jpg"), value)
