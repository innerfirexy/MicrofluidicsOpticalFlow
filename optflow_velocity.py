import cv2
from decord import VideoReader, cpu
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import PIL


def get_samples(input_video, output_dir: str = './sampled_frames', num_samples: int = 15):
    vr = VideoReader(input_video, ctx = cpu(0))
    # frame count
    print('Total # of frames: ', len(vr))

    file_name = os.path.basename(input_video)
    output_path = os.path.join(output_dir, os.path.splitext(file_name)[0])
    print(output_path)

    sample_every = len(vr) // num_samples
    count = 0
    for i in tqdm(range(len(vr))):
        if i % sample_every == 0 and i > 0:
            frame = vr[i]
            # print(type(frame)) # decord.ndarray.NDArray
            frame = frame.asnumpy()
            # print(type(frame))
            img = PIL.Image.fromarray(frame)
            image_name = output_path + f'_{count+1}.png'
            img.save(image_name)
            count += 1
        if count > 14:
            break
    # print(frame.shape)


def compute_mags(input_video, 
    output_dir: str = './sampled_histograms', 
    num_plots: int = 15):
    """
    Return:
    magnitudes
    """
    vr = VideoReader(input_video, ctx = cpu(0))
    plot_every = len(vr) // num_plots
    file_name = os.path.basename(input_video)
    output_path = os.path.join(output_dir, os.path.splitext(file_name)[0])

    cap = cv2.VideoCapture(input_video)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    frame_count = 0
    plt_count = 0
    magnitudes = []
    while cap.isOpened():
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            break

        # compute and plot
        if frame_count % plot_every == 0 and frame_count > 0:
            plt_count += 1
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            np.save(output_path + f'_mag{plt_count}.npy', mag)

            magnitudes.append(mag)
            # plt.hist(mag, bins=10)
            # plt_name = output_path + f'_{plt_count}.png'
            # plt.savefig(plt_name)

        prev = next
        frame_count += 1
        # cv2.imshow('frame2',rgb)
        sys.stdout.write(f'\r{frame_count}/{len(vr)} frames processed')
        sys.stdout.flush()
    
    cap.release()
    mag_filename = output_path + '_magnitudes.pkl'
    with open(mag_filename, 'wb') as f:
        pickle.dump(magnitudes, f)


def analyze_mags(input_path):
    # input_path = '200x200 micrometer  channel B  0.72 ul per min  40x  01'
    # mags = []
    # for i in range(1, 16, 1):
    #     mag = np.load(input_path + f'_mag{i}.npy')
    #     mags.append(mag)
    mag_filename = input_path + '_magnitudes.pkl'
    if os.path.exists(mag_filename):
        mags = pickle.load(open(mag_filename, 'rb'))
        avg_mag = np.mean(list(map(
            lambda x: np.mean(x[np.logical_and(x != np.inf, x > 3.0)]), mags)))

    print(f'{input_path}, {avg_mag:.3f}')


def pipeline():
    groups = [
        # ('./Microfluidics Dataset/50 x 50/With Dextran', '50x50',
        # 'D', ['0.06', '0.045', '0.075']),

        # ('./Microfluidics Dataset/100 x 100/With Dextran', '100x100',
        # 'C', ['0.18 ', '0.24', '0.30']),

        # ('./Microfluidics Dataset/200 x 200/With Dextran', '200x200',
        # 'B', ['0.72 ', '0.96 ', '1.2 ']),

        ('./Microfluidics Dataset/23.9x83.5/with dextran', '23.9x83.5',
        'dextran', ['v1', 'v2', 'v3']),

        ('./Microfluidics Dataset/23.9x83.5/without dextran', '23.9x83.5',
        'without dextran', ['v1', 'v2', 'v3']),
    ]
    result_folder = './sampled_histograms'
    analyze_flag = True
    compute_flag = False 

    for group in groups:
        video_folder, diameter, channel, flow_rates = group
        for rate in flow_rates:
            trials = ['01', '02', '03', '04']
            for t in trials:
                if diameter == '23.9x83.5':
                    input_name = f'{channel} {rate} {t}'
                else:
                    input_name = f'{diameter} micrometer  channel {channel}  {rate}ul per min  40x  {t}'

                if compute_flag:
                    input_path = os.path.join(video_folder, input_name + '.mp4')
                    print('computing ', input_path)
                    # print(os.path.exists(input_path))
                    compute_mags(input_path, output_dir=result_folder, num_plots=15)
                    print()
                if analyze_flag:
                    input_path = os.path.join(result_folder, input_name)
                    analyze_mags(input_path)


if __name__ == "__main__":
    # assert len(sys.argv) > 1
    # input_video = sys.argv[1]
    # compute_mags(input_video)
    # Time elapse for 1 video file: cpu 6:59.37 total
    pipeline()