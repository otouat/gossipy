
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import csv
from datetime import datetime
from typing import List, Dict, Tuple
from gossipy import LOG
from gossipy.topology import display_topology

def plot(report):
    fig, axs = plt.subplots(4, figsize=(10, 20))  # Increased figsize to accommodate the additional plot
    
    # Plot train and test accuracy for each node
    node_colors = plt.cm.get_cmap('tab10', len(report.get_accuracy()))

    for node_id, train_test_acc in report.get_accuracy().items():
        rounds = []
        train_accs = []
        test_accs = []
        for round_num, acc_list in train_test_acc:
            rounds.append(round_num)
            train_accs.append(acc_list['train'])
            test_accs.append(acc_list['test'])
        axs[0].plot(rounds, train_accs, 'o', label=f'Node {node_id} (Train)', linestyle='dashed', color=node_colors(node_id))
        axs[0].plot(rounds, test_accs, 'x', label=f'Node {node_id} (Test)', linestyle='solid', color=node_colors(node_id))
    
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Train and Test Accuracy per Node')
    #axs[0].legend(loc='upper left')
    axs[0].grid(True)
    
    # Plot mean MIA vulnerability each round
    mean_mia_vulnerability_per_round = [np.mean([mia['loss_mia'] for round_num, mia in mia_list]) for mia_list in report.get_mia_vulnerability().values()]
    rounds = range(1, len(mean_mia_vulnerability_per_round) + 1)
    axs[1].plot(rounds, mean_mia_vulnerability_per_round, label='Mean MIA Vulnerability', color='green')
    
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('MIA Vulnerability')
    axs[1].set_title('Mean MIA Vulnerability per Round')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot average generalization error
    avg_gen_error_per_round = []
    train_acc_list = []
    test_acc_list = []
    for round_num in range(1, len(mean_mia_vulnerability_per_round) + 1):
        for node_id, train_test_acc in report.get_accuracy().items():
            for r, acc_list in train_test_acc:
                if r == round_num:
                    train_acc_list.append(acc_list['train'])
                    test_acc_list.append(acc_list['test'])
        if train_acc_list and test_acc_list:
            avg_train_acc = np.mean(train_acc_list)
            avg_test_acc = np.mean(test_acc_list)
            gen_error = (avg_train_acc - avg_test_acc) / (avg_train_acc + avg_test_acc)
            avg_gen_error_per_round.append(gen_error)
        else:
            avg_gen_error_per_round.append(np.nan)
    
    axs[2].plot(rounds, avg_gen_error_per_round, label='Average Generalization Error', color='orange')
    axs[2].set_xlabel('Round')
    axs[2].set_ylabel('Generalization Error')
    axs[2].set_title('Average Generalization Error per Round')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot MIA vulnerability over Generalization Error
    axs[3].scatter(avg_gen_error_per_round, mean_mia_vulnerability_per_round, label='MIA Vulnerability over Generalization Error', color='blue')
    axs[3].set_xlabel('Average Generalization Error')
    axs[3].set_ylabel('Mean MIA Vulnerability')
    axs[3].set_title('MIA Vulnerability over Generalization Error')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    return fig

def log_results(Simul, report, topology):
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

    # Save diagrams
    fig = get_fig_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")
    fig2 = plot(report)
    fig3 = display_topology(topology)
    diagrams = {
        'Overall_gossipy_results': fig,
        'Overall_test_results': fig2,
        "Topology": fig3
    }
    for name, fig in diagrams.items():
        fig.savefig(f"{new_folder_path}/{name}.png")

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