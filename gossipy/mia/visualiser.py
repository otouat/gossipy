import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Read the CSV file
df = pd.read_csv(r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results\Exp_n#5\mia_results.csv")

    # Extract unique nodes
nodes = df['Node'].unique()

# Define the color map
node_colors = plt.cm.get_cmap('tab10', len(nodes))

# Plotting all four graphs together
fig, axs = plt.subplots(2, 2, figsize=(15, 10))


avg_train_acc = df.groupby('Round')['Train Accuracy'].mean()
avg_local_test_acc = df.groupby('Round')['Local Test Accuracy'].mean()
avg_global_test_acc = df.groupby('Round')['Global Test Accuracy'].mean()
std_train_acc = df.groupby('Round')['Train Accuracy'].std()
std_local_test_acc = df.groupby('Round')['Local Test Accuracy'].std()
std_global_test_acc = df.groupby('Round')['Global Test Accuracy'].std()

axs[0, 0].plot(avg_train_acc.index, avg_train_acc,'b-', label='Accuracy on train set')
axs[0, 0].fill_between(avg_train_acc.index, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='b', alpha=0.2)
axs[0, 0].plot(avg_local_test_acc.index, avg_local_test_acc, 'r--', label='Accuracy on local test set')
axs[0, 0].fill_between(avg_local_test_acc.index, avg_local_test_acc  - std_local_test_acc, avg_local_test_acc + std_local_test_acc, color='r', alpha=0.2)
axs[0, 0].plot(avg_global_test_acc.index, avg_global_test_acc, 'g--', label='Accuracy on global test set')
axs[0, 0].fill_between(avg_global_test_acc.index, avg_global_test_acc  - std_global_test_acc, avg_global_test_acc + std_global_test_acc, color='g', alpha=0.2)
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Calculating and plotting the average generalisation error over the epochs
gen_errors = (avg_train_acc - avg_local_test_acc) / (avg_train_acc + avg_local_test_acc)
std_gen_errors = (std_train_acc - std_local_test_acc) / (std_train_acc - std_local_test_acc)

axs[0, 1].plot(avg_train_acc.index, gen_errors)
#axs[0, 1].fill_between(avg_train_acc.index, gen_errors - std_gen_errors, gen_errors + std_gen_errors, color='b', alpha=0.2)
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Average Generalization Error')
axs[0, 1].set_title('Average Generalization Error over Epochs')
axs[0, 1].grid(True)

# Calculating the average MIA vulnerability at each round
avg_mia_loss = df.groupby('Round')['Loss MIA'].mean()
avg_mia_entropy = df.groupby('Round')['Entropy MIA'].mean()
std_mia_loss = df.groupby('Round')['Loss MIA'].std()
std_mia_entropy = df.groupby('Round')['Entropy MIA'].std()

axs[1, 0].plot(avg_mia_loss.index, avg_mia_loss,'b-', label='Average MIA Loss')
#axs[1, 0].fill_between(avg_mia_loss.index, avg_mia_loss - std_mia_loss, avg_mia_loss + std_mia_loss, color='b', alpha=0.2)
axs[1, 0].plot(avg_mia_entropy.index, avg_mia_entropy, 'r--', label='Average MIA Entropy')
#axs[1, 0].fill_between(avg_mia_entropy.index, avg_mia_entropy  - std_mia_entropy, avg_mia_entropy + std_mia_entropy, color='r', alpha=0.2)
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