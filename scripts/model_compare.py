import subprocess
import re
from termcolor import colored

model_scripts = [
    ('Logistic Regression', 'scripts/model_logistic_regression.py'),
    # ('SVM', 'scripts/model_svm.py'),
    ('Random Forest', 'scripts/model_random_forest.py'),
    ('XGBoost', 'scripts/model_xgboost.py'),
]

results = []

# Regex to extract metrics
metrics_re = re.compile(r"Accuracy: ([0-9.]+).*?Precision: ([0-9.]+).*?Recall: ([0-9.]+).*?F1 Score: ([0-9.]+).*?ROC-AUC: ([0-9.]+)", re.DOTALL)

for name, script in model_scripts:
    print(colored(f'Running {name}...', 'cyan'))
    proc = subprocess.run(['python3', script], capture_output=True, text=True)
    output = proc.stdout
    match = metrics_re.search(output)
    if match:
        acc, prec, rec, f1, roc = map(float, match.groups())
        results.append((name, acc, prec, rec, f1, roc))
    else:
        results.append((name, 'ERR', 'ERR', 'ERR', 'ERR', 'ERR'))
    print(colored(output, 'yellow'))

# Print summary table
print(colored('\nModel Comparison Summary:', 'green', attrs=['bold']))
header = f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}"
print(colored(header, 'magenta', attrs=['bold']))
for row in results:
    name, acc, prec, rec, f1, roc = row
    if acc != 'ERR':
        acc_c = colored(f"{acc:.4f}", 'cyan')
        prec_c = colored(f"{prec:.4f}", 'yellow')
        rec_c = colored(f"{rec:.4f}", 'blue')
        f1_c = colored(f"{f1:.4f}", 'green')
        roc_c = colored(f"{roc:.4f}", 'red')
    else:
        acc_c = prec_c = rec_c = f1_c = roc_c = colored('ERR', 'red', attrs=['bold'])
    print(f"{name:<20} {acc_c:<10} {prec_c:<10} {rec_c:<10} {f1_c:<10} {roc_c:<10}") 