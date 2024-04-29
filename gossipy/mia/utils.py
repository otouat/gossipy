
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import csv
from datetime import datetime
from typing import List, Dict, Tuple
from gossipy import LOG
from gossipy.topology import display_topology
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def log_results(Simul, report, topology, message, model_name, dataset_name):
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
        params_file.write(f"Models: {model_name}\n")
        params_file.write(f"Dataset: {dataset_name}\n")
        params_file.write(f"Message: {message}\n")

    # Save combined MIA vulnerability and accuracy
    combined_file_path = f"{new_folder_path}/mia_results.csv"
    with open(combined_file_path, 'w', newline='') as combined_file:
        writer = csv.writer(combined_file)
        writer.writerow(['Node', 'Round', 'Loss MIA', 'Entropy MIA', 'Train Accuracy', 'Test Accuracy'])
        
        for node_id, mia_vulnerabilities in report.get_mia_vulnerability().items():
            accuracies = report.get_accuracy()[node_id] if node_id in report.get_accuracy() else []
            
            for round_number, (mia_round, acc_round) in enumerate(zip(mia_vulnerabilities, accuracies), 1):
                mia_dict = mia_round[1]
                accuracy_dict = acc_round[1] if acc_round else {'train': None, 'test': None}
                writer.writerow([
                    node_id, 
                    round_number, 
                    mia_dict['loss_mia'], 
                    mia_dict['entropy_mia'], 
                    accuracy_dict['train'],
                    accuracy_dict['test']
                ])
    
    # Update the experiment number tracker file
    with open(exp_tracker_file, 'w') as file:
        file.write(str(experiment_number))
    
    # Save diagrams
    fig = get_fig_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")
    fig2 = plot(combined_file_path)
    if topology is not None:  # Modified condition to check if topology is not None
        fig3 = display_topology(topology)
        diagrams = {
            'Overall_gossipy_results': fig,
            'Overall_test_results': fig2,
            "Topology": fig3
        }
    else:
        diagrams = {
            'Overall_gossipy_results': fig,
            'Overall_test_results': fig2
        }
    for name, fig in diagrams.items():
        fig.savefig(f"{new_folder_path}/{name}.png")


def plot(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

        # Extract unique nodes
    nodes = df['Node'].unique()

    # Define the color map
    node_colors = plt.cm.get_cmap('tab10', len(nodes))

    # Plotting all four graphs together
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))


    avg_train_acc = df.groupby('Round')['Train Accuracy'].mean()
    avg_test_acc = df.groupby('Round')['Test Accuracy'].mean()
    std_train_acc = df.groupby('Round')['Train Accuracy'].std()
    std_test_acc = df.groupby('Round')['Test Accuracy'].std()

    axs[0, 0].plot(avg_train_acc.index, avg_train_acc,'b-', label='Average MIA Loss')
    axs[0, 0].fill_between(avg_train_acc.index, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='b', alpha=0.2)
    axs[0, 0].plot(avg_test_acc.index, avg_test_acc, 'r--', label='Average MIA Entropy')
    axs[0, 0].fill_between(avg_test_acc.index, avg_test_acc  - std_test_acc, avg_test_acc + std_test_acc, color='r', alpha=0.2)
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Train and Test Accuracy for Each Node over Epochs')
    axs[0, 0].grid(True)

    # Calculating and plotting the average generalisation error over the epochs
    gen_errors = (avg_train_acc - avg_test_acc) / (avg_train_acc + avg_test_acc)
    std_gen_errors = (std_train_acc - std_test_acc) / (std_train_acc - std_test_acc)

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
    return fig

def get_fig_evaluation(evals: List[List[Dict]],
                    title: str="Untitled plot") -> None:

    if not evals or not evals[0] or not evals[0][0]: return
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(111)
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu: float = np.mean(evs, axis=0)
        std: float = np.std(evs, axis=0)
        plt.fill_between(range(1, len(mu)+1), mu-std, mu+std, alpha=0.2)
        plt.title(title)
        plt.xlabel("cycle")
        plt.ylabel("metric value")
        plt.plot(range(1, len(mu)+1), mu, label=k)
        LOG.info(f"{k}: {mu[-1]:.2f}")
    ax.legend(loc="lower right")
    #plt.show()

    return fig
