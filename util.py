# Most code taken and adapted from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def _plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        ax = plt.subplot()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

            
def plot_confusion_matrix(y_true, y_pred, classes):

    np.set_printoptions(precision=2)
    fig = plt.figure(figsize=(10, 4))

    # Plot non-normalized confusion matrix
    ax = plt.subplot(121)
    _plot_confusion_matrix(y_true, y_pred, classes, ax=ax,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ax = plt.subplot(122)
    _plot_confusion_matrix(y_true, y_pred, classes, ax=ax, normalize=True,
                          title='Normalized confusion matrix')

    fig.tight_layout()
    
    return fig



from pathlib import Path



def _find_and_replace_string_in_file(file, old_string, new_string):
    # https://pythonexamples.org/python-replace-string-in-file/
    fin = open(file, "rt")
    data = fin.read()
    data = data.replace(old_string, new_string)
    fin.close()

    fin = open(file, "wt")
    fin.write(data)
    fin.close()
    
def make_metafiles_local(colab_googledrive_directory, local_googledrive):

    for filename in Path('mlruns').glob('**/meta.yaml'):

        colab_googledrive_directory_uri = 'file://' + f'{colab_googledrive_directory}'.replace(' ', '%20')
        _find_and_replace_string_in_file(filename, colab_googledrive_directory_uri, local_googledrive)
    
    return 'Success, paths in meta.yaml have been adapted to local', f'{colab_googledrive_directory_uri} is now {local_googledrive}'