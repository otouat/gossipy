import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import csv
from typing import List, Dict
from gossipy import LOG
import pandas as pd
from matplotlib.lines import Line2D

def log_results(Simul, report, wandb, message=""):
    base_folder_path = os.path.join(os.getcwd(), "results")
    exp_tracker_file = os.path.join(base_folder_path, "exp_number.txt")

    # Read the last experiment number and increment it
    if os.path.exists(exp_tracker_file):
        with open(exp_tracker_file, 'r') as file:
            experiment_number = int(file.read().strip()) + 1
    else:
        experiment_number = 1

    # Create new subfolder
    new_folder_path = f"{base_folder_path}/Exp_n#{experiment_number}"
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Log file path for experiment parameters
    params_file_path = f"{new_folder_path}/simulation_params.log"

    # Log experiment parameters
    with open(params_file_path, 'w') as params_file:
        params_file.write(f"Experiment Number: {experiment_number}\n")
        params_file.write(f"Protocol: {type(Simul).__name__}\n")
        params_file.write(f"Timestamp: {datetime.now()}\n")
        params_file.write(f"Total Nodes: {Simul.n_nodes}\n")
        params_file.write(f"Total Rounds: {Simul.n_rounds}\n")
        params_file.write(f"Message: {message}\n")

    # Save combined MIA vulnerability and accuracy
    combined_file_path = f"{new_folder_path}/mia_results.csv"
    with open(combined_file_path, 'w', newline='') as combined_file:
        writer = csv.writer(combined_file)
        writer.writerow(['Node', 'Round', 'Loss MIA', 'Entropy MIA', 'Marginalized Loss MIA', 'Marginalized Entropy MIA', 'Train Accuracy', 'Local Test Accuracy', 'Global Test Accuracy'])

        all_node_ids = set(report.get_mia_vulnerability(False).keys()).union(set(report.get_accuracy(True).keys()))

        for node_id in all_node_ids:
            mia_vulnerabilities = report.get_mia_vulnerability(False).get(node_id, [])
            marginalized_mia_vulnerabilities = report.get_mia_vulnerability(True).get(node_id, [])
            local_accuracies = report.get_accuracy(True).get(node_id, [])
            global_accuracies = report.get_accuracy(False).get(node_id, [])

            # Determine the number of rounds based on the maximum length of the lists
            num_rounds = max(len(mia_vulnerabilities), len(local_accuracies), len(global_accuracies))

            for round_number in range(1, num_rounds + 1):
                # Initialize dictionaries
                mia_vulnerabilities_dict = {'loss_mia': None, 'entropy_mia': None}
                marginalized_mia_vulnerabilities_dict = {'loss_mia': None, 'entropy_mia': None}
                local_accuracy_dict = {'train': None, 'test': None}
                global_accuracy_dict = {'test': None}

                # Extract MIA vulnerabilities if they exist
                if round_number - 1 < len(mia_vulnerabilities):
                    mia_vulnerabilities_dict = mia_vulnerabilities[round_number - 1][1]
                if round_number - 1 < len(marginalized_mia_vulnerabilities):
                    marginalized_mia_vulnerabilities_dict = marginalized_mia_vulnerabilities[round_number - 1][1]

                # Extract local and global accuracy metrics
                if round_number - 1 < len(local_accuracies):
                    local_accuracy_dict = local_accuracies[round_number - 1][1]
                if round_number - 1 < len(global_accuracies):
                    global_accuracy_dict = global_accuracies[round_number - 1][1]

                # Write row to CSV
                writer.writerow([
                    node_id,
                    round_number,
                    mia_vulnerabilities_dict.get('loss_mia', None),
                    mia_vulnerabilities_dict.get('entropy_mia', None),
                    marginalized_mia_vulnerabilities_dict.get('loss_mia', None),
                    marginalized_mia_vulnerabilities_dict.get('entropy_mia', None),
                    local_accuracy_dict.get('train', None),
                    local_accuracy_dict.get('test', None),
                    global_accuracy_dict.get('test', None)
                ])

                wandb.log({
                    "node": node_id,
                    "round": round_number,
                    "loss_mia": mia_vulnerabilities_dict.get('loss_mia', None),
                    "entropy_mia": mia_vulnerabilities_dict.get('entropy_mia', None),
                    "marginalized_loss_mia": marginalized_mia_vulnerabilities_dict.get('loss_mia', None),
                    "marginalized_entropy_mia": marginalized_mia_vulnerabilities_dict.get('entropy_mia', None),
                    "train_accuracy": local_accuracy_dict.get('train', None),
                    "local_test_accuracy": local_accuracy_dict.get('test', None),
                    "global_test_accuracy": global_accuracy_dict.get('test', None)
                })

    # Update the experiment number tracker file
    with open(exp_tracker_file, 'w') as file:
        file.write(str(experiment_number))
    print("Experiment parameters logged successfully.")
    
    # Save diagrams
    print("Generating diagrams...")
    fig = get_fig_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")
    fig2 = plot(combined_file_path)
    diagrams = {
        'Overall_gossipy_results': fig,
        'Overall_test_results': fig2,
    }

    for name, fig in diagrams.items():
        fig_path = f"{new_folder_path}/{name}.png"
        fig.savefig(fig_path)
        print(f"Diagram saved: {fig_path}")

    print("Diagrams saved successfully.")

def plot(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract unique nodes
    nodes = df['Node'].unique()

    # Define the color map
    node_colors = plt.cm.get_cmap('tab10', len(nodes))

    # Plotting all four graphs together
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    print(df)

    avg_train_acc = df.groupby('Round')['Train Accuracy'].mean()
    avg_local_test_acc = df.groupby('Round')['Local Test Accuracy'].mean()
    avg_global_test_acc = df.groupby('Round')['Global Test Accuracy'].mean()
    std_train_acc = df.groupby('Round')['Train Accuracy'].std()
    std_local_test_acc = df.groupby('Round')['Local Test Accuracy'].std()
    std_global_test_acc = df.groupby('Round')['Global Test Accuracy'].std()

    rounds = range(1, len(avg_train_acc) + 1)

    axs[0, 0].plot(rounds, avg_train_acc, 'C1-', label='Accuracy on train set')
    axs[0, 0].fill_between(rounds, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='C1', alpha=0.2)
    axs[0, 0].plot(rounds, avg_local_test_acc, 'C2--', label='Accuracy on local test set')
    axs[0, 0].fill_between(rounds, avg_local_test_acc - std_local_test_acc, avg_local_test_acc + std_local_test_acc, color='C2', alpha=0.2)
    axs[0, 0].plot(rounds, avg_global_test_acc, 'C3--', label='Accuracy on global test set')
    axs[0, 0].fill_between(rounds, avg_global_test_acc - std_global_test_acc, avg_global_test_acc + std_global_test_acc, color='C3', alpha=0.2)
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Calculating and plotting the average generalization error over the epochs
    gen_errors = (avg_train_acc - avg_local_test_acc) / (avg_train_acc + avg_local_test_acc)
    std_gen_errors = (std_train_acc - std_local_test_acc) / (std_train_acc + std_local_test_acc)

    axs[0, 1].plot(rounds, gen_errors, "C1")
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

    axs[1, 0].plot(rounds, avg_mia_entropy, 'C1-', label='Average MIA Entropy')
    axs[1, 0].plot(rounds, avg_mar_mia_entropy, 'C2--', label='Average Marginalized MIA Entropy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Average MIA Vulnerability')
    axs[1, 0].set_title('Average MIA Vulnerability over Epochs')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Including the MIA entropy vulnerability of node 0
    axs[1, 0].plot(rounds, avg_mia_entropy, 'C0-', label='Average MIA Entropy Node 0')
    axs[1, 0].legend()

    # Calculating the average MIA vulnerability over the generalization errors
    axs[1, 1].scatter(gen_errors, avg_mia_entropy, c='C1', label='Average MIA Entropy')
    axs[1, 1].scatter(gen_errors, avg_mar_mia_entropy, c='C2', label='Average Marginalized MIA Entropy')
    axs[1, 1].set_xlabel('Average Generalization Error')
    axs[1, 1].set_ylabel('Average MIA Vulnerability')
    axs[1, 1].set_title('Average MIA Vulnerability over Generalization Errors')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    return fig

def get_fig_evaluation(evals: List[List[Dict]], title: str="Untitled plot") -> None:
    if not evals or not evals[0] or not evals[0][0]:
        return
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(111)
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu = np.mean(evs, axis=0)
        std = np.std(evs, axis=0)
        plt.fill_between(range(1, len(mu)+1), mu-std, mu+std, alpha=0.2)
        plt.title(title)
        plt.xlabel("cycle")
        plt.ylabel("metric value")
        plt.plot(range(1, len(mu)+1), mu, label=k)
        LOG.info(f"{k}: {mu[-1]:.2f}")
    ax.legend(loc="lower right")
    return fig

