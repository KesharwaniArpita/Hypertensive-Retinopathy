def plot_accuracy(history, model_name):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)
    test_acc = history.history['val_accuracy'][-1] * np.ones(len(train_acc))

    # Plot the accuracy curves
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.plot(epochs, test_acc, 'g-', label='Test Accuracy')
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
