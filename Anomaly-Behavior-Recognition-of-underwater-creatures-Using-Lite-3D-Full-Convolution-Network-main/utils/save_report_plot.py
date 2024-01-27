import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, \
    precision_recall_curve, roc_curve, auc, accuracy_score
import seaborn as sns
import os


def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    # plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    # plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    nb_epoch = len(accuracy)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], accuracy[i], val_loss[i], val_accuracy[i]))


def getPrecision_Recall_F1score(Y_true, Y_score, target_names):
    precision = dict()
    recall = dict()
    average_precision = dict()
    threshold = dict()
    f1_score = dict()
    # macro_p = 0
    # macro_r = 0
    # macro_t = 0
    for i, target_name in enumerate(target_names):
        # print(i)
        precision[target_name], recall[target_name], threshold[target_name] = precision_recall_curve(Y_true[:, i], Y_score[:, i])
        # macro_p += precision[target_name]
        # macro_r += recall[target_name]
        # macro_t += threshold[target_name]
        average_precision[target_name] = average_precision_score(Y_true[:, i], Y_score[:, i])
        f1_score[target_name] = 2 * ((precision[target_name] * recall[target_name]) / (precision[target_name] + recall[target_name]))
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(Y_true.ravel(), Y_score.ravel())
    # precision["macro"], recall["macro"], threshold["macro"] = macro_p / (i + 1), macro_r / (i + 1), macro_t / (i + 1)
    average_precision["micro"] = average_precision_score(Y_true, Y_score, average="micro")
    # average_precision["macro"] = average_precision_score(Y_true, Y_score)
    f1_score["micro"] = 2 * ((precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"]))
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))
    return precision, recall, average_precision, f1_score, threshold


def savePRCurve(filename, precision, recall, average_precision, target_names, colors):
    plt.figure(figsize=(7, 8))
    # f, (ax1) = plt.subplots(figsize=(10, 8), nrows=1)
    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    for i, color in zip(target_names, colors):
        # l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    # colors.insert(0, "gold")
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(1, .5), prop=dict(size=14))
    # plt.legend(lines, labels, loc=1, prop=dict(size=14), fontsize="xx-small")
    fig.savefig(filename, bbox_inches='tight')


def saveAvgPRCurve01(filename, precision, recall, average_precision, target_names, colors):
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color='navy', lw=2)
    lines.append(l)
    # labels.append('macro-average Precision-recall (area = {0:0.2f})'
    #               ''.format(average_precision["macro"]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to average')
    plt.legend(lines, labels, loc=(1, .5), prop=dict(size=14))
    fig.savefig(filename, bbox_inches='tight')


def saveF1Curve(filename, f1_score, threshold, average_precision, target_names, colors):
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    l, = plt.plot(np.r_[threshold["micro"], 1], f1_score["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average F1-score = {0:0.2f}'
                  ''.format(average_precision["micro"]))
    for i, color in zip(target_names, colors):
        l, = plt.plot(np.r_[threshold[i], 1], f1_score[i], color=color, lw=2)
        lines.append(l)
        labels.append('F1-score for class {0} ({1:0.2f})'
                      ''.format(i, average_precision[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('threshold')
    plt.ylabel('f1-score')
    plt.title('Extension of F1-score curve')
    plt.legend(lines, labels, loc=(1, .5), prop=dict(size=14))
    fig.savefig(filename, bbox_inches='tight')


def saveROCCurve(filename, fpr, tpr, roc_auc, target_names, colors, lw=2):
    plt.figure(figsize=(7, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(target_names, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc=(1, .5))
    fig.savefig(filename, bbox_inches='tight')


def getRocAuc(Y_true, Y_score, target_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, target_name in enumerate(target_names):
        fpr[target_name], tpr[target_name], _ = roc_curve(Y_true[:, i], Y_score[:, i])
        roc_auc[target_name] = auc(fpr[target_name], tpr[target_name])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[target_name] for target_name in target_names]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i, target_name in enumerate(target_names):
        mean_tpr += np.interp(all_fpr, fpr[target_name], tpr[target_name])

    # Finally average it and compute AUC
    mean_tpr /= len(target_names)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def saveConfussionMatrix(filename, C2):
    sns.set()
    # sns.set_context({"figure.figsize": (8, 8)})
    f, (ax2) = plt.subplots(figsize=(10, 8), nrows=1)
    heatmap = sns.heatmap(np.array(C2), annot=True, fmt="d", linewidths=0.2, annot_kws={'size': 10, 'weight': 'bold'}, square=True)
    ax2.set_title('sns_heatmap_confusion_matrix')
    ax2.set_xlabel('Pred')
    ax2.set_ylabel('True')
    # f = heatmap.get_figure()
    f.savefig(filename, bbox_inches='tight')


def saveAllReportandPlot(model, X_test, Y_test, y, output):
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'blue', 'green', 'yellow', "purple"]
    predictions = model.predict(X_test)
    out= np.argmax(predictions, axis=1)
    # out = model.predict_classes(X_test)
    Y_score = model.predict(X_test)
    y_true = [np.argmax(x) for x in Y_test]
    c_m_y_true = [y[i] for i in y_true]
    c_m_out = [y[i] for i in out]
    C2 = confusion_matrix(c_m_y_true, c_m_out, labels=y)
    saveConfussionMatrix(f'{output}/confusion_matrix.png', C2)
    A = classification_report(y_true, out, target_names=y, digits=4)
    print(A)
    file = open(f"{output}/classification_data.txt", 'w')
    file.write(A)
    file.close()
    precision, recall, average_precision, f1_score, threshold = getPrecision_Recall_F1score(Y_test, Y_score, y)
    savePRCurve(f'{output}/PR.png', precision, recall, average_precision, y, colors)
    saveAvgPRCurve01(f'{output}/avg_PR.png', precision, recall, average_precision, y, colors)
    saveF1Curve(f'{output}/F1score.png', f1_score, threshold, average_precision, y, colors)
    fpr, tpr, roc_auc = getRocAuc(Y_test, Y_score, y)
    lw = 2
    saveROCCurve(f'{output}/ROC.png', fpr, tpr, roc_auc, y, colors, lw)
    print(f"saved in path {output}")
