import cv2
from decord import VideoReader, cpu
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import PIL
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Microfluidics ex-vivo data analysis')
parser.add_argument('--analyze', type=int, default=1)
parser.add_argument('--compute', type=int, default=1)
parser.add_argument('--save', type=str, default='')
args = parser.parse_args()


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
    ret, frame = cap.read()
    old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    frame_count = 0
    plt_count = 0
    magnitudes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute and plot
        if frame_count % plot_every == 0 and frame_count > 0:
            plt_count += 1
            flow = cv2.calcOpticalFlowFarneback(old_frame, new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            np.save(output_path + f'_mag{plt_count}.npy', mag)

            magnitudes.append(mag)
            # plt.hist(mag, bins=10)
            # plt_name = output_path + f'_{plt_count}.png'
            # plt.savefig(plt_name)

        old_frame = new_frame
        frame_count += 1
        # cv2.imshow('frame2',rgb)
        sys.stdout.write(f'\r{frame_count}/{len(vr)} frames processed')
        sys.stdout.flush()
    
    cap.release()
    mag_filename = output_path + '_magnitudes.pkl'
    with open(mag_filename, 'wb') as f:
        pickle.dump(magnitudes, f)


def analyze_mags(input_path, plot=False):
    mag_filename = input_path + '_magnitudes.pkl'
    mean_mags = None
    if os.path.exists(mag_filename):
        mags = pickle.load(open(mag_filename, 'rb'))
        mean_mags = np.mean(list(map(
            lambda x: np.mean(x[np.logical_and(x != np.inf, x > .5)]), mags)))
        # print(f'{input_path}, {mean_mags:.3f}')
        if plot:
            plt.hist(mags[0], bins=10)
            plt_name = input_path + '_avgmag.png'
            plt.savefig(plt_name)
    else:
        print(f'{mag_filename} does not exist.')
    
    return mean_mags


def analyze_vars(input_path, plot=False):
    mag_filename = input_path + '_magnitudes.pkl'
    mean_vars = None
    if os.path.exists(mag_filename):
        mags = pickle.load(open(mag_filename, 'rb'))
        variances = []
        for m in mags:
            m_valid = m[np.logical_and(m != np.inf, m > .5)]
            vari = np.std(m_valid)
            variances.append(vari)
        mean_vars = np.mean(np.array(variances))
        # print(f'{input_path}, {mean_vars:.3f}')
    else:
        print(f'{mag_filename} does not exist.')
    
    return mean_vars


def unit_test():
    input_path = './sampled_histograms/50x50 micrometer  channel D  0.06ul per min  40x  01'
    analyze_vars(input_path)


def pipeline():
    groups = [
        # With dextran
        # ('./Microfluidics Dataset/50 x 50/With Dextran', '50x50',
        # 'D', ['0.06', '0.045', '0.075']),

        # ('./Microfluidics Dataset/100 x 100/With Dextran', '100x100',
        # 'C', ['0.18 ', '0.24', '0.30']),

        # ('./Microfluidics Dataset/200 x 200/With Dextran', '200x200',
        # 'B', ['0.72 ', '0.96 ', '1.2 ']),

        ('./Microfluidics Dataset/23.9x83.5/with dextran', '23.9x83.5',
        'dextran', ['v1', 'v2', 'v3']),

        # Without dextran
        # ('./Microfluidics Dataset/50 x 50/Without Dextran', '50x50',
        # 'D', ['0.06', '0.045', '0.075']),

        # ('./Microfluidics Dataset/100 x 100/Without Dextran', '100x100',
        # 'C', ['0.18 ', '0.24', '0.30']),

        # ('./Microfluidics Dataset/200 x 200/Without Dextran', '200x200',
        # 'B', ['0.72 ', '0.96 ', '1.2 ']),

        ('./Microfluidics Dataset/23.9x83.5/without dextran', '23.9x83.5',
        'without dextran', ['v1', 'v2', 'v3'])
    ]
    result_folder = './sampled_histograms'

    analysis_results = []
    for group in groups:
        video_folder, diameter, channel, flow_rates = group
        for rate in flow_rates:
            trials = ['01', '02', '03', '04']
            for t in trials:
                if diameter == '23.9x83.5':
                    input_name = f'{channel} {rate} {t}'
                else:
                    input_name = f'{diameter} micrometer  channel {channel}  {rate}ul per min  40x  {t}'

                if args.compute == 1:
                    input_path = os.path.join(video_folder, input_name + '.mp4')
                    print('computing ', input_path)
                    # print(os.path.exists(input_path))
                    compute_mags(input_path, output_dir=result_folder, num_plots=15)
                    print()
                if args.analyze == 1:
                    input_path = os.path.join(result_folder, input_name)
                    mean_mags = analyze_mags(input_path)
                    mean_vars = analyze_vars(input_path)
                    analysis_results.append((input_path, mean_mags, mean_vars))
    
    if args.save:
        data = pd.DataFrame.from_records(analysis_results, columns=['name', 'mean', 'var'])
        data.to_csv(args.save, index=False)


if __name__ == "__main__":
    pipeline()
    # unit_test()