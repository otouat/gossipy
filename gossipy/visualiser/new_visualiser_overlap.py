import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files for both experiments
df1 = pd.read_csv(r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results\Exp_n#75\mia_results.csv")
df2 = pd.read_csv(r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results\Exp_n#76\mia_results.csv")  # Update the path for the second experiment

# Extract unique nodes for both experiments
nodes1 = df1['Node'].unique()
nodes2 = df2['Node'].unique()

fts=14

# Define the color map
node_colors = plt.cm.get_cmap('tab10', max(len(nodes1), len(nodes2)))

# Plotting all four graphs together
mpl.style.use('seaborn-v0_8')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Calculate statistics for the first experiment
avg_train_acc1 = df1.groupby('Round')['Train Accuracy'].mean()
avg_local_test_acc1 = df1.groupby('Round')['Local Test Accuracy'].mean()
avg_global_test_acc1 = df1.groupby('Round')['Global Test Accuracy'].mean()
std_train_acc1 = df1.groupby('Round')['Train Accuracy'].std()
std_local_test_acc1 = df1.groupby('Round')['Local Test Accuracy'].std()
std_global_test_acc1 = df1.groupby('Round')['Global Test Accuracy'].std()

# Calculate statistics for the second experiment
avg_train_acc2 = df2.groupby('Round')['Train Accuracy'].mean()
avg_local_test_acc2 = df2.groupby('Round')['Local Test Accuracy'].mean()
avg_global_test_acc2 = df2.groupby('Round')['Global Test Accuracy'].mean()
std_train_acc2 = df2.groupby('Round')['Train Accuracy'].std()
std_local_test_acc2 = df2.groupby('Round')['Local Test Accuracy'].std()
std_global_test_acc2 = df2.groupby('Round')['Global Test Accuracy'].std()

# Determine the minimum number of rounds between both experiments
min_rounds = min(len(avg_train_acc1), len(avg_train_acc2))
rounds = range(1, min_rounds + 1)

# Plot the first experiment results (up to min_rounds)
axs[0, 0].plot(rounds, avg_train_acc1[:min_rounds], 'C1:', label='Exp 1: Accuracy on train set')
axs[0, 0].fill_between(rounds, avg_train_acc1[:min_rounds] - std_train_acc1[:min_rounds], avg_train_acc1[:min_rounds] + std_train_acc1[:min_rounds], color='C1', alpha=0.2)
axs[0, 0].plot(rounds, avg_local_test_acc1[:min_rounds], 'C1-', label='Exp 1: Accuracy on local test set')
axs[0, 0].fill_between(rounds, avg_local_test_acc1[:min_rounds] - std_local_test_acc1[:min_rounds], avg_local_test_acc1[:min_rounds] + std_local_test_acc1[:min_rounds], color='C2', alpha=0.2)
axs[0, 0].plot(rounds, avg_global_test_acc1[:min_rounds], 'C1--', label='Exp 1: Accuracy on global test set')
axs[0, 0].fill_between(rounds, avg_global_test_acc1[:min_rounds] - std_global_test_acc1[:min_rounds], avg_global_test_acc1[:min_rounds] + std_global_test_acc1[:min_rounds], color='g', alpha=0.2)

# Plot the second experiment results (up to min_rounds)
axs[0, 0].plot(rounds, avg_train_acc2[:min_rounds], 'C2:', label='Exp 2: Accuracy on train set')
axs[0, 0].fill_between(rounds, avg_train_acc2[:min_rounds] - std_train_acc2[:min_rounds], avg_train_acc2[:min_rounds] + std_train_acc2[:min_rounds], color='C1', alpha=0.1)
axs[0, 0].plot(rounds, avg_local_test_acc2[:min_rounds], 'C2-', label='Exp 2: Accuracy on local test set')
axs[0, 0].fill_between(rounds, avg_local_test_acc2[:min_rounds] - std_local_test_acc2[:min_rounds], avg_local_test_acc2[:min_rounds] + std_local_test_acc2[:min_rounds], color='C2', alpha=0.1)
axs[0, 0].plot(rounds, avg_global_test_acc2[:min_rounds], 'C2--', label='Exp 2: Accuracy on global test set')
axs[0, 0].fill_between(rounds, avg_global_test_acc2[:min_rounds] - std_global_test_acc2[:min_rounds], avg_global_test_acc2[:min_rounds] + std_global_test_acc2[:min_rounds], color='g', alpha=0.1)

axs[0, 0].set_xlabel('Epochs', fontsize=fts)
axs[0, 0].set_ylabel('Accuracy', fontsize=fts)
axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs', fontsize=fts)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Calculating and plotting the average generalization error over the epochs for both experiments
gen_errors1 = (avg_train_acc1 - avg_local_test_acc1) / (avg_train_acc1 + avg_local_test_acc1)
std_gen_errors1 = (std_train_acc1 - std_local_test_acc1) / (std_train_acc1 + std_local_test_acc1)
gen_errors2 = (avg_train_acc2 - avg_local_test_acc2) / (avg_train_acc2 + avg_local_test_acc2)
std_gen_errors2 = (std_train_acc2 - std_local_test_acc2) / (std_train_acc2 + std_local_test_acc2)

axs[0, 1].plot(rounds, gen_errors1[:min_rounds], 'C1-', label='Exp 1: Generalization Error')
axs[0, 1].plot(rounds, gen_errors2[:min_rounds], 'C2-', label='Exp 2: Generalization Error')
axs[0, 1].set_xlabel('Epochs', fontsize=fts)
axs[0, 1].set_ylabel('Average Generalization Error', fontsize=fts)
axs[0, 1].set_title('Average Generalization Error over Epochs', fontsize=fts)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Calculating the average MIA vulnerability at each round for both experiments
avg_mia_loss1 = df1.groupby('Round')['Loss MIA'].mean()
avg_mia_entropy1 = df1.groupby('Round')['Entropy MIA'].mean()
std_mia_loss1 = df1.groupby('Round')['Loss MIA'].std()
std_mia_entropy1 = df1.groupby('Round')['Entropy MIA'].std()

avg_mia_loss2 = df2.groupby('Round')['Loss MIA'].mean()
avg_mia_entropy2 = df2.groupby('Round')['Entropy MIA'].mean()
std_mia_loss2 = df2.groupby('Round')['Loss MIA'].std()
std_mia_entropy2 = df2.groupby('Round')['Entropy MIA'].std()
'''
avg_mar_mia_loss1 = df1.groupby('Round')['Marginalized Loss MIA'].mean()
avg_mar_mia_entropy1 = df1.groupby('Round')['Marginalized Entropy MIA'].mean()
std_mar_mia_loss1 = df1.groupby('Round')['Marginalized Loss MIA'].std()
std_mar_mia_entropy1 = df1.groupby('Round')['Marginalized Entropy MIA'].std()

avg_mar_mia_loss2 = df2.groupby('Round')['Marginalized Loss MIA'].mean()
avg_mar_mia_entropy2 = df2.groupby('Round')['Marginalized Entropy MIA'].mean()
std_mar_mia_loss2 = df2.groupby('Round')['Marginalized Loss MIA'].std()
std_mar_mia_entropy2 = df2.groupby('Round')['Marginalized Entropy MIA'].std()
'''
# Scatter plots for the first experiment (up to min_rounds)
axs[1, 0].scatter(avg_global_test_acc1[:min_rounds], avg_mia_entropy1[:min_rounds], label='Exp 1: Average MIA Entropy', color='C1')
#axs[1, 0].scatter(avg_global_test_acc1[:min_rounds], avg_mar_mia_entropy1[:min_rounds], label='Exp 1: Average Marginalized MIA Entropy', color='C3', marker='x')
# Scatter plots for the second experiment (up to min_rounds)
axs[1, 0].scatter(avg_global_test_acc2[:min_rounds], avg_mia_entropy2[:min_rounds], label='Exp 2: Average MIA Entropy', color='C2')
#axs[1, 0].scatter(avg_global_test_acc2[:min_rounds], avg_mar_mia_entropy2[:min_rounds], label='Exp 2: Average Marginalized MIA Entropy', color='C4', marker='x')

axs[1, 0].set_xlabel('Global Test Accuracy', fontsize=fts)
axs[1, 0].set_ylabel('Average MIA Vulnerability', fontsize=fts)
axs[1, 0].set_title('Average MIA Vulnerability over Global Test Accuracy', fontsize=fts)
axs[1, 0].legend()
axs[1, 0].grid(True)

# Scatter plots for the first experiment (up to min_rounds)
axs[1, 1].scatter(gen_errors1[:min_rounds], avg_mia_entropy1[:min_rounds], label='Exp 1: Average MIA Entropy', color='C1')
#axs[1, 1].scatter(gen_errors1[:min_rounds], avg_mar_mia_entropy1[:min_rounds], label='Exp 1: Average Marginalized MIA Entropy', color='C3', marker='x')
# Scatter plots for the second experiment (up to min_rounds)
axs[1, 1].scatter(gen_errors2[:min_rounds], avg_mia_entropy2[:min_rounds], label='Exp 2: Average MIA Entropy', color='C2')
#axs[1, 1].scatter(gen_errors2[:min_rounds], avg_mar_mia_entropy2[:min_rounds], label='Exp 2: Average Marginalized MIA Entropy', color='C4', marker='x')

axs[1, 1].set_xlabel('Average Generalization Error', fontsize=fts)
axs[1, 1].set_ylabel('Average MIA Vulnerability', fontsize=fts)
axs[1, 1].set_title('Average MIA Vulnerability over Generalization Errors', fontsize=fts)
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
