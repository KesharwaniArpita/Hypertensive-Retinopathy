def plot_accuracy_overall(overall_accuracy_history,overall_val_accuracy_history):
    overall_acc = np.mean(overall_accuracy_history, axis=0)
    overall_val_acc = np.mean(overall_val_accuracy_history, axis=0)  # Modified: Use correct variable for validation accuracy
    plt.plot(overall_acc, label='Training Accuracy')
    plt.plot(overall_val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Overall Training and Validation Accuracy')
    plt.legend()
    plt.show()
