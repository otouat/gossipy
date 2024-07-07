import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory
base_dir = r"C:\Users\jezek\OneDrive\Documents\Python\Djack\gossipy\results"

# Initialize lists to store the generalization errors and MIA vulnerabilities for federated, static, and dynamic experiments
gen_errors_federated = []
mia_entropy_federated = []
gen_errors_static = []
mia_entropy_static = []
gen_errors_dynamic = []
mia_entropy_dynamic = []

exp_to_check = [53, 54]
#exp_to_check = range (16, 21)
parameter = "peer sampling period"
paramA = "peer sampling period: 1"
paramB = "peer sampling period: 3"
paramC = "peer sampling period: 5"

def get_marker(parameter):
    marker_dict = {
        paramA: 's',
        paramB: 'v',
        paramC: 'D'
        # Add more mappings if needed for other parameters
    }
    return marker_dict.get(parameter, 'v')  # Default to 'o' if parameter is not found

# Loop through each experiment from 1 to 160
for exp_num in exp_to_check:
    #logging.info(f"Processing experiment #{exp_num}")
    
    # Define the paths for the CSV and parameter files
    mia_results_path = os.path.join(base_dir, f"Exp_n#{exp_num}", "mia_results.csv")
    param_file_path = os.path.join(base_dir, f"Exp_n#{exp_num}", "simulation_params.log")
    
    # Check if the files exist
    if not os.path.exists(mia_results_path):
        #logging.warning(f"mia_results.csv not found for experiment #{exp_num}")
        continue
    if not os.path.exists(param_file_path):
        #logging.warning(f"simulation_params.log not found for experiment #{exp_num}")
        continue
    
    # Read the MIA results CSV file
    try:
        df = pd.read_csv(mia_results_path)
    except Exception as e:
        #logging.error(f"Error reading mia_results.csv for experiment #{exp_num}: {e}")
        continue
    
    # Limit to rounds after the 50th round
    df = df[df['Round'] > 50]
    
    # Determine if the experiment is federated, static, or dynamic from the parameter file
    try:
        with open(param_file_path, 'r') as param_file:
            params = param_file.read()
            is_federated = 'AttackFederatedSimulator' in params or 'MIAFederatedSimulator' in params
            is_static = 'MIAGossipSimulator' in params or 'AttackGossipSimulator' in params
            is_dynamic = 'MIADynamicGossipSimulator' in params or 'AttackDynamicGossipSimulator' in params
            
            # Extract the desired parameter for marker differentiation
            parameter = paramA if paramA in params else paramB if paramB in params else paramC if paramC in params else 'Other'
    except Exception as e:
        #logging.error(f"Error reading simulation_params.log for experiment #{exp_num}: {e}")
        continue
    
    # Check for the correct column names
    train_acc_col = 'Local Train Accuracy' if 'Local Train Accuracy' in df.columns else 'Train Accuracy'
    test_acc_col = 'Local Test Accuracy' if 'Local Test Accuracy' in df.columns else 'Test Accuracy'
    
    if train_acc_col not in df.columns or test_acc_col not in df.columns:
        #logging.error(f"Required columns not found in mia_results.csv for experiment #{exp_num}")
        continue
    
    # Calculate metrics for rounds after the 50th round
    try:
        avg_train_acc = df.groupby('Round')[train_acc_col].mean()
        avg_local_test_acc = df.groupby('Round')[test_acc_col].mean()
        gen_error = (avg_train_acc - avg_local_test_acc) / (avg_train_acc + avg_local_test_acc)
        avg_mia_entropy = df.groupby('Round')['Entropy MIA'].mean()
        
        max_mia_round = avg_mia_entropy.idxmax()
        max_mia_entropy = avg_mia_entropy[max_mia_round]
        max_gen_error = gen_error[max_mia_round]
    except Exception as e:
        #logging.error(f"Error calculating metrics for experiment #{exp_num}: {e}")
        continue
    if is_federated:
        print(f"Experiment #{exp_num} is federated")
    if 0.6 <= max_mia_entropy <= 0.8 and 0.2 <= max_gen_error <= 0.6:
        print(f"Experiment #{exp_num} has max MIA entropy {max_mia_entropy:.2f} and gen_error {max_gen_error:.2f}")

    # Store the results in the appropriate lists
    marker = get_marker(parameter)  # Get marker based on the parameter
    
    if is_federated:
        gen_errors_federated.append((gen_error.mean(), marker))  # Store average gen_error
        mia_entropy_federated.append(avg_mia_entropy.mean())  # Store average MIA entropy
    elif is_static:
        gen_errors_static.append((gen_error.mean(), marker))  # Store average gen_error
        mia_entropy_static.append(avg_mia_entropy.mean())  # Store average MIA entropy
    elif is_dynamic:
        gen_errors_dynamic.append((gen_error.mean(), marker))  # Store average gen_error
        mia_entropy_dynamic.append(avg_mia_entropy.mean())  # Store average MIA entropy

# Plot the results
logging.info("Plotting results")
try:
    mpl.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 6))
    
    # Helper function to plot data points with different markers
    def plot_with_marker(data, values, color, label):
        for (gen_error, marker), mia_entropy in zip(data, values):
            parameter_name = paramA if marker == 's' else paramB if marker == "v" else paramC # Get parameter name based on marker
            plt.scatter(gen_error, mia_entropy, color=color, marker=marker, label=f'{label}: {parameter_name}', s=250)
    
    plot_with_marker(gen_errors_federated, mia_entropy_federated, 'C1', 'Federated')
    plot_with_marker(gen_errors_static, mia_entropy_static, 'C2', 'Static')
    plot_with_marker(gen_errors_dynamic, mia_entropy_dynamic, 'C3', 'Dynamic')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(by_label.values(), by_label.keys(), fontsize='large', labelspacing=1.0)


    plt.xlabel('\nAverage Generalization Error', color='C0', fontsize=15)  # Update xlabel
    plt.ylabel('Average MIA Accuracy\n', color='C0', fontsize=15)  # Update ylabel
    plt.title('Average MIA Accuracy over Generalization Errors\n', color='C0', fontsize=17)  # Update title
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("Plotting completed successfully")
except Exception as e:
    logging.error(f"Error plotting results: {e}")
