import os
import time
import multiprocessing as mp
from pathlib import Path
import cv2


def load_video(file_path):
    start_time = time.time()

    try:
        cap = cv2.VideoCapture(str(file_path))
        frames = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = int(fps / 10)  # Get every Nth frame to achieve 10fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep frames at 10fps
            if frame_count % sample_interval == 0:
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        print(f"took {time.time() - start_time:.2f} seconds to execute")

        return len(frames)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return 0

def process_videos(folder_path):
    # Get list of mp4 files
    video_files = list(Path(folder_path).glob('*.mp4'))
    
    # Create pool of workers
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    # Process files in parallel
    results = pool.map(load_video, video_files)
    
    # Clean up
    pool.close()
    pool.join()
    
    # Print results
    total_frames = sum(results)
    print(f"Processed {len(video_files)} videos at 10fps")
    print(f"Total frames (at 10fps): {total_frames}")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time/len(video_files):.2f} seconds")

if __name__ == '__main__':
    # Replace with your video folder path
    video_folder = "/tmp/zverev/datasets/vggsound-copy/video"
    process_videos(video_folder)

