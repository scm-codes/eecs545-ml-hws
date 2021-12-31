# EECS 545 Fall 2021
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, stats=[], name='CNN'):
        self.stats = stats
        self.name = name
        self.axes = self.make_cnn_training_plot()

    def make_cnn_training_plot(self):
        """
        Runs the setup for an interactive matplotlib graph that logs the loss and accuracy
        """
        print('Setting up interactive graph...')
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(self.name + ' Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        return axes

    def log_cnn_training(self, epoch):
        """
        Logs the validation accuracy and loss to the terminal
        """
        valid_acc, valid_loss, train_acc, train_loss = self.stats[-1]
        print('Epoch {}'.format(epoch))
        print('\tValidation Loss: {}'.format(valid_loss))
        print('\tValidation Accuracy: {}'.format(valid_acc))
        print('\tTrain Loss: {}'.format(train_loss))
        print('\tTrain Accuracy: {}'.format(train_acc))

    def update_cnn_training_plot(self, epoch):
        """
        Updates the training plot with a new data point for loss and accuracy
        """
        xrange = range(epoch - len(self.stats) + 1, epoch + 1)
        self.axes[0].plot(xrange, [s[0] for s in self.stats], linestyle='--', marker='o', color='b')
        self.axes[0].plot(xrange, [s[2] for s in self.stats], linestyle='--', marker='o', color='r')
        self.axes[1].plot(xrange, [s[1] for s in self.stats], linestyle='--', marker='o', color='b')
        self.axes[1].plot(xrange, [s[3] for s in self.stats], linestyle='--', marker='o', color='r')
        self.axes[0].legend(['Validation', 'Train'])
        self.axes[1].legend(['Validation', 'Train'])
        plt.pause(0.00001)

    def save_cnn_training_plot(self):
        """
        Saves the training plot to a file
        """
        plt.savefig(self.name + '_training_plot.png', dpi=200)

    def hold_training_plot(self):
        """
        Keep the program alive to display the training plot
        """
        plt.ioff()
        plt.show()
