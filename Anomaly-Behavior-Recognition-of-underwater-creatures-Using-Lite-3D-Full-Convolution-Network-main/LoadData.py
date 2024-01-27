import os
import numpy as np
import tqdm


def loaddata(img_dir, img3d, nclass, result_dir, color=False, skip=True):
    files = os.listdir(img_dir)
    X = []
    labels = []
    labellist = []
    pbar = tqdm(total=len(files))
    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(img_dir, filename)
        for sample_files in os.listdir(name):
            img_file_path = os.path.join(name, sample_files)
            img_files = [f"{img_file_path}/{x}" for x in os.listdir(img_file_path)]
            label = filename

            if label not in labellist:

                if len(labellist) >= nclass:
                    continue

                labellist.append(label)

            labels.append(label)

            X.append(img3d.img3d(img_files, color=color, skip=skip))

    pbar.close()

    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels



