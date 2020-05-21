import os
import glob
import argparse
import cv2

from .feature_selection import select_features

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir')
    parser.add_argument('--features_filepath')
    parser.add_argument('--save_dir', default='/home/thad/Desktop/verify')
    args = parser.parse_args()

    FEATURES_TO_EXTRACT = ['left_hand_x', 'left_hand_y',
                           'left_hand_w', 'left_hand_h',
                           'right_hand_x', 'right_hand_y',
                           'right_hand_w', 'right_hand_h']

    image_filepaths_glob = os.path.join(args.images_dir, '*png')
    image_filepaths = glob.glob(image_filepaths_glob)

    features_df = select_features(args.features_filepath, FEATURES_TO_EXTRACT,
                                  center_on_face=False, scale=1)

    w, h, _ = cv2.imread(image_filepaths[0]).shape
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #video_writer = cv2.VideoWriter(args.mp4_filepath, fourcc, 20, (w,h))

    for i, image_filepath in enumerate(image_filepaths):

        #filename = image_filepath.split('/')[-1]
        filename = f'frame_{i:03d}.png'
        print(filename)
        save_image_filepath = os.path.join(args.save_dir, filename)

        image = cv2.imread(image_filepath)
        x1, y1, w1, h1 = features_df.loc[i, FEATURES_TO_EXTRACT[:4]].values
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        w1 = int(w1 * w)
        h1 = int(h1 * h)
        x2, y2, w2, h2 = features_df.loc[i, FEATURES_TO_EXTRACT[4:]].values
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        w2 = int(w2 * w)
        h2 = int(h2 * h)
        cv2.rectangle(image, (x1, h-y1), (x1+w1, h-(y1+h1)), (255, 0, 0), 2)
        cv2.rectangle(image, (x2, h-y2), (x2+w2, h-(y2+h2)), (0, 0, 255), 2)
        cv2.imwrite(save_image_filepath, image)
        #video_writer.write(image)
