import re
import matplotlib.pyplot as plt

def plot_mia_vulnerability(mia_accuracy, gen_error):
    fig = plt.figure()
    plt.scatter(gen_error, mia_accuracy, label='MIA Vulnerability over Generalization Error', color='blue')
    plt.xlabel('Generalization Error')
    plt.ylabel('MIA Vulnerability')
    plt.title('MIA Vulnerability over Generalization Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    fig2 = plt.figure()
    epoch = list(range(1, len(mia_accuracy) + 1))
    plt.plot(epoch, mia_accuracy, label='MIA Vulnerability per epoch', color='green')
    plt.xlabel('Epoch n°')
    plt.ylabel('MIA Vulnerability')
    plt.title('MIA Vulnerability per epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return fig, fig2

# Read the log file
with open('results\Exp n°32\simulation_results.log', 'r') as file:
    log_content = file.read()

# Extract MIA vulnerability values
mia_accuracy = re.findall(r'MIA Vulnerability: \[(.*?)\]', log_content, re.DOTALL)
mia_accuracy = [float(x) for x in mia_accuracy[0].split(", ")]

# Extract generalization error values
gen_error = re.findall(r'Gen Error: \[\[(.*?)\]\]', log_content, re.DOTALL)
gen_error = [[float(x) for x in re.findall(r'\d+\.\d+', row)] for row in gen_error[0].split(", ")]
gen_error = [item for sublist in gen_error for item in sublist]

print("mia_accuracy: ", mia_accuracy)
print(len(mia_accuracy))
print("gen_error: ", gen_error)
print(len(gen_error))
# Plot the graphs+
size = len(mia_accuracy)
size = 90
plot_mia_vulnerability(mia_accuracy[-size:], gen_error[-size:])
