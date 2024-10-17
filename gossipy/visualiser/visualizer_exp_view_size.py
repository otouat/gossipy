import pandas as pd
import matplotlib.pyplot as plt
import os
import re  # We'll use regex to extract the number of neighbors

def plot_all_experiments(base_dir, target_neighbors=None):
    # Initialize a dictionary to store the data for all experiments based on neighbor count
    experiment_data = {}

    for exp_num in range(1, 500):  # Assuming you have 500 experiments
        file_path = os.path.join(base_dir, f"Exp_n#{exp_num}", "mia_results.csv")
        param_file_path = os.path.join(base_dir, f"Exp_n#{exp_num}", "simulation_params.log")

        # Check if files exist
        if not os.path.exists(file_path) or not os.path.exists(param_file_path):
            continue

        # Extract number of neighbors from the params file
        num_neighbors = None
        try:
            with open(param_file_path, 'r') as param_file:
                params_content = param_file.read()
                
                # Check if the experiment is using a static or dynamic topology
                if 'MIAGossipSimulator' in params_content or 'AttackGossipSimulator' in params_content:
                    topology_type = 'static'
                elif 'MIADynamicGossipSimulator' in params_content or 'AttackDynamicGossipSimulator' in params_content:
                    topology_type = 'dynamic'

                # Use regex to find the number of neighbors from the relevant line
                print(params_content)
                match = re.search(r'neighbors (\d+)', params_content)
                print(match)
                if match:
                    num_neighbors = match.group(1)  # Extract the number of neighbors
        except Exception as e:
            continue  # Skip if there was an error reading params or no neighbor count found
        
        if num_neighbors is None:
            continue  # Skip if no neighbor count was found
        
        # Filter by target_neighbors if specified
        if target_neighbors is not None and (int(num_neighbors) != 2):
            if (int(num_neighbors) != 35):
                continue  # Skip this experiment if it doesn't match the target number of neighbors
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Store data based on the number of neighbors and topology type
        key = (num_neighbors, topology_type)
        if key not in experiment_data:
            experiment_data[key] = []
        experiment_data[key].append(df)
        print(experiment_data)
    # Now plot all the experiments based on the number of neighbors
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for (num_neighbors, topology_type), dfs in experiment_data.items():
        if len(dfs) == 0:
            continue  # Skip if no data for this neighbor count

        # Combine all dataframes for the current neighbor count
        combined_df = pd.concat(dfs)

        # Group by round and calculate mean and standard deviation for each metric
        avg_train_acc = combined_df.groupby('Round')['Train Accuracy'].mean()
        avg_local_test_acc = combined_df.groupby('Round')['Local Test Accuracy'].mean()
        avg_global_test_acc = combined_df.groupby('Round')['Global Test Accuracy'].mean()
        std_train_acc = combined_df.groupby('Round')['Train Accuracy'].std()
        std_local_test_acc = combined_df.groupby('Round')['Local Test Accuracy'].std()
        std_global_test_acc = combined_df.groupby('Round')['Global Test Accuracy'].std()

        rounds = range(1, len(avg_train_acc) + 1)

        # Plot Train and Test Accuracy
        axs[0, 0].plot(rounds, avg_train_acc, label=f'Train Accuracy (neighbors={num_neighbors}, topology={topology_type})')
        axs[0, 0].fill_between(rounds, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, alpha=0.2)
        axs[0, 0].plot(rounds, avg_local_test_acc, '--', label=f'Local Test Accuracy (neighbors={num_neighbors}, topology={topology_type})')
        axs[0, 0].fill_between(rounds, avg_local_test_acc - std_local_test_acc, avg_local_test_acc + std_local_test_acc, alpha=0.2)
        axs[0, 0].plot(rounds, avg_global_test_acc, '-.', label=f'Global Test Accuracy (neighbors={num_neighbors}, topology={topology_type})')
        axs[0, 0].fill_between(rounds, avg_global_test_acc - std_global_test_acc, avg_global_test_acc + std_global_test_acc, alpha=0.2)
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_title('Train and Test Accuracy')
        axs[0, 0].grid(True)

        # Plot Generalization Error
        gen_errors = (avg_train_acc - avg_local_test_acc) / (avg_train_acc + avg_local_test_acc)
        axs[0, 1].plot(rounds, gen_errors, label=f'Gen Error (neighbors={num_neighbors}, topology={topology_type})')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Generalization Error')
        axs[0, 1].set_title('Generalization Error over Epochs')
        axs[0, 1].grid(True)

        # MIA Vulnerability
        avg_mia_entropy = combined_df.groupby('Round')['Entropy MIA'].mean()
        axs[1, 0].plot(rounds, avg_mia_entropy, label=f'MIA Entropy (neighbors={num_neighbors}, topology={topology_type})')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('MIA Vulnerability')
        axs[1, 0].set_title('MIA Vulnerability over Epochs')
        axs[1, 0].grid(True)

        # Scatter plot of MIA vulnerability vs. Generalization Error
        axs[1, 1].scatter(gen_errors, avg_mia_entropy, label=f'MIA Entropy (neighbors={num_neighbors}, topology={topology_type})')
        axs[1, 1].set_xlabel('Generalization Error')
        axs[1, 1].set_ylabel('MIA Vulnerability')
        axs[1, 1].set_title('MIA Vulnerability vs Generalization Error')
        axs[1, 1].grid(True)

    # Add legends and finalize the layout
    for ax in axs.flat:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('all_topologies_plot_n150_neigh2_localstep5.pdf', format='pdf', bbox_inches='tight')

# Example usage
base_dir = r"/home/otouat/git_repositories/gossipy/results"
plot_all_experiments(base_dir, target_neighbors=2)
