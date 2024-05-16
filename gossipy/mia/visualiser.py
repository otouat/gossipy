import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results\Exp_n#43\mia_results.csv")

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

rounds = range(1, len(avg_train_acc) + 1)

axs[0, 0].plot(rounds, avg_train_acc, 'b-', label='Accuracy on train set')
axs[0, 0].fill_between(rounds, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='b', alpha=0.2)
axs[0, 0].plot(rounds, avg_local_test_acc, 'r--', label='Accuracy on local test set')
axs[0, 0].fill_between(rounds, avg_local_test_acc - std_local_test_acc, avg_local_test_acc + std_local_test_acc, color='r', alpha=0.2)
axs[0, 0].plot(rounds, avg_global_test_acc, 'g--', label='Accuracy on global test set')
axs[0, 0].fill_between(rounds, avg_global_test_acc - std_global_test_acc, avg_global_test_acc + std_global_test_acc, color='g', alpha=0.2)
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Calculating and plotting the average generalization error over the epochs
gen_errors = (avg_train_acc - avg_local_test_acc) / (avg_train_acc + avg_local_test_acc)
std_gen_errors = (std_train_acc - std_local_test_acc) / (std_train_acc + std_local_test_acc)

axs[0, 1].plot(rounds, gen_errors)
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Average Generalization Error')
axs[0, 1].set_title('Average Generalization Error over Epochs')
axs[0, 1].grid(True)

# Calculating the average MIA vulnerability at each round
avg_mia_loss = df.groupby('Round')['Loss MIA'].mean()
avg_mia_entropy = df.groupby('Round')['Entropy MIA'].mean()
std_mia_loss = df.groupby('Round')['Loss MIA'].std()
std_mia_entropy = df.groupby('Round')['Entropy MIA'].std()

avg_mar_mia_loss = df.groupby('Round')['Marginalized Loss MIA'].mean()
avg_mar_mia_entropy = df.groupby('Round')['Marginalized Entropy MIA'].mean()
std_mar_mia_loss = df.groupby('Round')['Marginalized Loss MIA'].std()
std_mar_mia_entropy = df.groupby('Round')['Marginalized Entropy MIA'].std()

axs[1, 0].plot(rounds, avg_mia_entropy, 'b-', label='Average MIA Entropy')
axs[1, 0].plot(rounds, avg_mar_mia_entropy, 'r--', label='Average Marginalized MIA Entropy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Average MIA Vulnerability')
axs[1, 0].set_title('Average MIA Vulnerability over Epochs')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Calculating the average MIA vulnerability over the generalization errors
axs[1, 1].scatter(gen_errors, avg_mia_entropy, label='Average MIA Entropy')
axs[1, 1].scatter(gen_errors, avg_mar_mia_entropy, label='Average Marginalized MIA Entropy')
axs[1, 1].set_xlabel('Average Generalization Error')
axs[1, 1].set_ylabel('Average MIA Vulnerability')
axs[1, 1].set_title('Average MIA Vulnerability over Generalization Errors')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()