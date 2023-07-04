def plot_loss(history, model_name):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)
    

    # Plot the loss curves
    plt.plot(epochs, train_acc, 'b-', label='Training Loss')
    plt.plot(epochs, val_acc, 'r-', label='Validation Loss')
    
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
