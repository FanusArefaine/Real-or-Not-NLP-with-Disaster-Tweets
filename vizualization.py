
import matplotlib.pyplot as plt 
import seaborn as sns

def distribution_plot(df, feature_name):
    plt.subplots(figsize=(10,6))

    """
        A function to plot distribution of a feature
        against the target values for comparison.

        feature_name: input name of the feature to plot 
        against the target value

        output of the function is a plot
    """

    

    sns.distplot(df[feature_name][df['target']==0], color='green', label='Not a disaster tweet')
    sns.distplot(df[feature_name][df['target']==1], color='blue', label='Disaster tweet')
    plt.title("Training Dataset " + feature_name + " Distribution")
    plt.legend()
    plt.show()



def plot_learning_curves(history):
    #plots training accuracy and loss 
    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])

    ax[0].legend(['train', 'validation'], loc = 'upper left')
    ax[1].legend(['train', 'validation'], loc = 'upper left')

    fig.suptitle("Model Accuracy", fontsize=14)

    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')

    return plt.show()

