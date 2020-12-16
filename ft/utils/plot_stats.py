from matplotlib import pyplot as plt
import numpy as np

def plot_save_results(records, save_dir, num_epochs, stage = 'Train'):

    losses = [each['loss'] for each in records]
    f1s = [each['f1'] for each in records]

    total_steps = len(records)
    losses_rm = np.array(losses)\
            .dot(np.triu(np.ones((total_steps, total_steps), dtype=int)))/(np.arange(total_steps)+1)
    f1s_rm = np.array(f1s)\
            .dot(np.triu(np.ones((total_steps, total_steps), dtype=int)))/(np.arange(total_steps)+1)

    plt.plot(losses, label='loss')
    plt.plot(losses_rm, label='loss_rm')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(stage + ' loss')
    plt.legend()
    plt.savefig(save_dir + stage + "_loss.png")
    plt.clf()
    plt.plot(f1s, label='f1')
    plt.plot(f1s_rm, label='f1_rm')
    plt.xlabel('steps')
    plt.ylabel('f1')
    plt.title(stage + ' f1')
    plt.legend()
    plt.savefig(save_dir + stage + "_f1.png")
    plt.clf()