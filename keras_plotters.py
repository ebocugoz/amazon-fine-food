#Functions to plot learning curves
import matplotlib.pyplot as plt

def prepare_standardplot(title, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_ylabel('categorical cross entropy')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plot_history(history, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(history['loss'], label = "training")
    ax1.plot(history['val_loss'], label = "validation")
    ax2.plot(history['accuracy'], label = "training")
    ax2.plot(history['val_accuracy'], label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig
def plot_multiple(histories,names,title):
  fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
  for history,name in zip(histories,names):
    ax1.plot(history['loss'], label = "training_"+name)
    ax1.plot(history['val_loss'], label = "validation_"+name)
    ax2.plot(history['accuracy'], label = "training_"+name)
    ax2.plot(history['val_accuracy'], label = "validation_"+name)
  finalize_standardplot(fig, ax1, ax2)
  return fig
  