import os
import numpy as np
import argparse
import multiprocessing


def process_file(params):
    input_f, target_fold = params
    ext_len = len(input_f.split("/")[-1].split(".")[-1])
    video_id = input_f.split("/")[-1][: -ext_len - 1]
    output_f_1 = os.path.join(target_fold, video_id + "_intermediate.wav")
    output_f_2 = os.path.join(target_fold, video_id + ".wav")

    # first resample audio and save to intermediate file
    os.system("ffmpeg -i {:s} -vn -ar 16000 {:s}".format(input_f, output_f_1))

    # then extract the first channel
    os.system("sox {:s} {:s} remix 1".format(output_f_1, output_f_2))

    # remove the intermediate file
    os.remove(output_f_1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Easy video feature extractor")
    parser.add_argument(
        "-input_file_list",
        type=str,
        default="sample_video_extract_list.csv",
        help="Should be a csv file of a single column, each row is the input video path.",
    )
    parser.add_argument(
        "-target_fold",
        type=str,
        default="./sample_audio/",
        help="The place to store the audio files.",
    )
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, delimiter=",", dtype=str)
    if not os.path.exists(args.target_fold):
        os.makedirs(args.target_fold)

    params = [(input_f, args.target_fold) for input_f in input_filelist]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(process_file, params)
