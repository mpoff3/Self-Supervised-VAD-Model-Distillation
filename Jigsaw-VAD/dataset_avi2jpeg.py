import argparse
import os
import cv2
from tqdm import tqdm
DATA_DIR = os.environ["VAD_DATASET_PATH"]


def frame_video(file):
    cap = cv2.VideoCapture(file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def main(dataset_name, force=False):
    final_dataset_path = os.path.join(DATA_DIR, f'{dataset_name}_frames')
    os.makedirs(final_dataset_path, exist_ok=force)
    for key in ['testing', 'training']:
        final_trainORtest_path = os.path.join(final_dataset_path, key)
        trainORtest_path = os.path.join(DATA_DIR, f'{dataset_name}_raw', key)
        os.makedirs(final_trainORtest_path, exist_ok=force)
        for video in tqdm(os.listdir(trainORtest_path)):
            if not video.endswith('.avi'):
                continue
            video_name = os.path.splitext(os.path.basename(video))[0]
            video_path = os.path.join(trainORtest_path, video)
            final_video_folder_path = os.path.join(final_trainORtest_path, video_name)
            os.makedirs(final_video_folder_path, exist_ok=force)
            for i, frame in enumerate(frame_video(video_path)):
                cv2.imwrite(os.path.join(final_video_folder_path, f"{i:04d}.jpg"), frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a .avi dataset to the structure required by the repo")
    parser.add_argument("--dataset_name", type=str, default='avenue', required=True)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    main(args.dataset_name, args.force)
