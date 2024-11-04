# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os.path
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def extract_frame(input_video_path, target_fold, extract_frame_num=10):
    # TODO: you can define your own way to extract video_id
    ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
    video_id = input_video_path.split('/')[-1][:-ext_len-1]
    
    # Check if all frames already exist
    all_frames_exist = True
    for i in range(extract_frame_num):
        frame_path = target_fold + '/frame_{:d}/'.format(i) + video_id + '.jpg'
        if not os.path.exists(frame_path):
            all_frames_exist = False
            break
            
    if all_frames_exist:
        print('All frames already exist for video: ' + video_id)
        return
        
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
    for i in range(extract_frame_num):
        frame_path = target_fold + '/frame_{:d}/'.format(i) + video_id + '.jpg'
        if os.path.exists(frame_path):
            continue
            
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        print('Extract frame {:d} from original frame {:d}, total video frame {:d} at frame rate {:d}.'.format(i, frame_idx, total_frame_num, int(fps)))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        # save in 'target_path/frame_{i}/video_id.jpg'
        if not os.path.exists(target_fold + '/frame_{:d}/'.format(i)):
            os.makedirs(target_fold + '/frame_{:d}/'.format(i))
        save_image(image_tensor, frame_path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Python script to extract frames from a video, save as jpgs.")
    parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
    parser.add_argument("-target_fold", type=str, default='./sample_frames/', help="The place to store the video frames.")
    args = parser.parse_args()

    # note the first row (header) is skipped
    input_filelist = np.loadtxt(args.input_file_list, dtype=str, delimiter=',')
    num_file = input_filelist.shape[0]
    print('Total {:d} videos are input'.format(num_file))
    import multiprocessing

    def process_video(file_id, video_path, target_fold):
        try:
            print('processing video {:d}: {:s}'.format(file_id, video_path))
            extract_frame(video_path, target_fold)
        except:
            print('error with ', video_path)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # Submit tasks to the pool
    for file_id in range(num_file):
        pool.apply_async(process_video, args=(file_id, input_filelist[file_id], args.target_fold))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()