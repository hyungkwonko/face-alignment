import face_alignment
import matplotlib.pyplot as plt
import collections
from skimage import io
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--indir', help='input data', type=str, required=True) # 'exp'
parser.add_argument('--device', help='device', type=str, choices=['cuda', 'cpu'], default='cuda')
parser.add_argument('--plot', help='plot', type=bool, default=True)
args = parser.parse_args()

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)
pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
            }

def get_landmark(indir, outdir, plot=False):

    input = io.imread(indir)
    pred = fa.get_landmarks(input)[-1]

    if type(pred) != type(None):
        if plot:
            save_image(input, outdir, pred)


def get_landmark_dir(indir, plot=False):

    preds = fa.get_landmarks_from_directory(indir)

    landmarks = []
    files = []

    for ix, i in enumerate(preds):
        print(f"[{ix + 1} / {len(preds)}] landmark detection running... >> {i}")
        if type(preds[i]) != type(None):
            pred = preds[i][0]

            # numpy save using reshape for landmark
            # [[1, 2], [3, 4]] --> [1, 2, 3, 4]
            # e.g., (x1, y1), (x2, y2) --> (x1, y1, x2, y2)
            landmarks.append(pred.reshape(-1))

            files.append(i)

            if plot:
                input = io.imread(i)
                outdir = i.replace('images', 'out')
                save_image(input, outdir, pred)

    npydir = indir.replace('images', 'npy')
    np.save(os.path.join(npydir, 'files.npy'), files)
    np.save(os.path.join(npydir, 'landmarks.npy'), landmarks)

    
def save_image(input, outdir, pred):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input)

    for pred_type in pred_types.values():
        ax.plot(pred[pred_type.slice, 0],
                pred[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    plt.savefig(outdir)


if __name__ == '__main__':

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

    indir = os.path.join('assets', args.indir, 'images')

    if os.path.isdir(indir):  
        print("[INFO] It is a directory")  
        get_landmark_dir(indir, plot=False)
    elif os.path.isfile(indir):
        print("[INFO] It is a file")
        outdir = indir.replace('images', 'out')
        get_landmark(indir, outdir, plot=True)
    else:
        print("No file detected!")