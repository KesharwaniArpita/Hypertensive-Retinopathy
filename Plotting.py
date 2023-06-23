import matplotlib.pyplot as plt

def plot_accuracy(history, model_name):
    # Get the accuracy values from the history object
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Get the number of epochs
    epochs = range(1, len(accuracy) + 1)

    # Plot the accuracy curves
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.plot(epochs, test_acc, 'g-', label='Test Accuracy')
    plt.title(model_name,' Accuracy - Training and Test Data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
