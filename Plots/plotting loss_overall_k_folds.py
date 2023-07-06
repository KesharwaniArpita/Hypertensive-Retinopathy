def plot_loss_overall(overall_loss_history, overall_val_loss_history):
    overall_loss = np.mean(overall_loss_history, axis=0)
    overall_val_loss = np.mean(overall_val_loss_history, axis=0)  # Modified: Use correct variable for validation loss
    plt.plot(overall_loss, label='Training Loss')
    plt.plot(overall_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overall Training and Validation Loss')
    plt.legend()
    plt.show()
