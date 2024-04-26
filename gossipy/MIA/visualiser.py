import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Read the CSV file
df = pd.read_csv(r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results\Exp_n#73\mia_results.csv")

# Extract unique nodes
nodes = df['Node'].unique()

# Define the color map
node_colors = plt.cm.get_cmap('tab10', len(nodes))

# Plotting all four graphs together
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plotting the accuracy of train and test for each node over the epochs
for node in nodes:
    node_df = df[df['Node'] == node]
    color = node_colors(node)
    axs[0, 0].plot(node_df['Round'], node_df['Train Accuracy'], 'o', label=f'Train Accuracy - Node {node}', linestyle='dashed', color=color)
    axs[0, 0].plot(node_df['Round'], node_df['Test Accuracy'], 'o', label=f'Test Accuracy - Node {node}', linestyle='solid', color=color)
custom_lines = [Line2D([0], [0], linestyle='dashed'),
                Line2D([0], [0], linestyle='solid',)]

axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs')
axs[0, 0].legend(custom_lines, ['Train Accuracy', 'Test Accuracy'])
axs[0, 0].grid(True)

# Calculating and plotting the average generalisation error over the epochs
avg_acc_train = df.groupby('Round')['Train Accuracy'].mean()
avg_acc_test = df.groupby('Round')['Test Accuracy'].mean()
gen_errors = (avg_acc_train - avg_acc_test) / (avg_acc_train + avg_acc_test)

axs[0, 1].plot(avg_acc_train.index, gen_errors)
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Average Generalization Error')
axs[0, 1].set_title('Average Generalization Error over Epochs')
axs[0, 1].grid(True)

# Calculating the average MIA vulnerability at each round
avg_mia_loss = df.groupby('Round')['Loss MIA'].mean()
avg_mia_entropy = df.groupby('Round')['Entropy MIA'].mean()

axs[1, 0].plot(avg_mia_loss.index, avg_mia_loss, label='Average MIA Loss')
axs[1, 0].plot(avg_mia_entropy.index, avg_mia_entropy, linestyle='--', label='Average MIA Entropy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Average MIA Vulnerability')
axs[1, 0].set_title('Average MIA Vulnerability over Epochs')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Calculating the average MIA vulnerability over the generalization errors
axs[1, 1].scatter(gen_errors, avg_mia_loss, label='Average MIA Loss')
axs[1, 1].scatter(gen_errors, avg_mia_entropy, label='Average MIA Entropy')
axs[1, 1].set_xlabel('Average Generalization Error')
axs[1, 1].set_ylabel('Average MIA Vulnerability')
axs[1, 1].set_title('Average MIA Vulnerability over Generalization Errors')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()