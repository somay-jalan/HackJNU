import argparse
import os
import matplotlib
matplotlib.use('AGG')
import numpy as np
from sklearn.model_selection import train_test_split
import imgto3d
from  tensorflow.keras.callbacks import ModelCheckpoint
from  tensorflow.keras.models import load_model
##
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session


sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
##
from utils.focal_loss import *
from utils import dataprocess as dp
from utils import save_report_plot as srp
from utils import models


def main():
    parser = argparse.ArgumentParser(
        description='3-D convolution networks for action recognition')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--images', type=str, default='dataset/mix_fishaction_screen_v04',
                        help='directory where images are stored')
    parser.add_argument('--nclass', type=int, default=7)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--model', type=str, default="lite3d")
    parser.add_argument('--loss', type=str, default="cross")
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mix_fishaction_screen_v04')
    parser.add_argument('--aug', type=bool, default=False)
    args = parser.parse_args()
    size = 64
    img_rows, img_cols, frames = size, size, args.depth
    channel = 3 if args.color else 1
    fname_npz = '{}_{}_{}_{}.npz'.format(
        args.dataset, args.nclass, args.depth, args.skip)

    img3d = imgto3d.Imgto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    data_set = args.dataset
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if "ucf" in data_set:
        nb_classes = int(data_set[3:])
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
        x, y = dp.loaddata(args.images, img3d, args.nclass, args.output, args.color, args.skip)
        X, Y = dp.dataPreprocess(x, y, img_rows, img_cols, frames, channel, nb_classes)
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    y = dp.loadCategory()
    if args.aug:
        output = f"{args.output}/aug/{args.model}_{args.iter}/{args.loss}"
    else:
        output = f"{args.output}/nonaug/{args.model}_{args.iter}/{args.loss}"
    if not os.path.isdir(output):
        os.makedirs(output)
    # Define model
    model = models.getModel(args, (X.shape[1:]), nb_classes)

    if args.aug:
        aug_X, aug_Y = dp.getMixAugData(X, Y)
    else:
        aug_X, aug_Y = X, Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        aug_X, aug_Y, test_size=0.2, random_state=43)
    print(f"X_train: {len(X_train)}, X_test: {len(X_test)}, Y_train: {len(Y_train)}, Y_test: {len(Y_test)}")
    ####################
    #1
    filepath="d_3dcnnmodel-{epoch:02d}-{val_accuracy:.2f}.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # 2
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.compat.v1.Session(config=config))
    ###############
    # Train model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True, callbacks=callbacks_list)
    model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    with open(os.path.join(output, f'{data_set}_{args.model}_{args.loss}_ar.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(output, f'{data_set}_{args.model}_{args.loss}_ar.hd5'))
    model.save('save_model_ar.hd5')

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    srp.plot_history(history, output)
    srp.save_history(history, output)
    srp.saveAllReportandPlot(model, X_test, Y_test, y, output)
    # _ = [os.remove(f"./{x}") for x in os.listdir("./") if x.endswith(".hd5")]


# def chechImage():

if __name__ == '__main__':
    main()
