import os
import re
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import pandas as pd

# Configuración
log_dir = "outputs/salud_mental/sinKG/"
file_pattern = "salud_mental{}_sinKG.log"
output_dir = "estadisticas/salud_mental/sinKG"
file_dev = "salud_mentalsinKG_evolucion_dev.png"
file_test = "salud_mentalsinKG_evolucion_test.png"
file_datos = "salud_mentalsinKG_datos.txt"
os.makedirs(output_dir, exist_ok=True)

# Expresiones regulares
acc_pattern = re.compile(r"Acc\. \(Correct/Total\): ([0-9.]+)")
time_pattern = re.compile(r"Tiempo total de ejecución: ([0-9.]+) segundos\.")

# Datos acumulados
final_accuracies = []
execution_times = []
epoch_dev_accs = []
epoch_test_accs = []

# Procesamiento de archivos
for i in range(1, 11):
    dev_accs = []
    test_accs = []
    final_test_acc = None
    time_taken = None
    current_eval = None  # Puede ser 'dev' o 'test'

    file_path = os.path.join(log_dir, file_pattern.format(i))
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            if "Start evaluation on dev dataset" in line:
                current_eval = "dev"
            elif "Start evaluation on test dataset" in line:
                current_eval = "test"

            acc_match = acc_pattern.search(line)
            if acc_match:
                acc = float(acc_match.group(1))
                if current_eval == "dev":
                    dev_accs.append(acc)
                elif current_eval == "test":
                    test_accs.append(acc)

            if "Final evaluation on the test dataset" in line:
                for offset in range(1, 15):
                    if j + offset < len(lines):
                        match = acc_pattern.search(lines[j + offset])
                        if match:
                            final_test_acc = float(match.group(1))
                            break

            if "Tiempo total de ejecución" in line:
                match = time_pattern.search(line)
                if match:
                    time_taken = float(match.group(1))

    final_accuracies.append(final_test_acc)
    execution_times.append(time_taken)
    epoch_dev_accs.append(dev_accs)
    epoch_test_accs.append(test_accs)

# Igualar longitud de listas
max_len = max(map(len, epoch_dev_accs))
for acc_list in epoch_dev_accs:
    acc_list.extend([np.nan] * (max_len - len(acc_list)))
for acc_list in epoch_test_accs:
    acc_list.extend([np.nan] * (max_len - len(acc_list)))

# Estadísticas por época (usando ddof=1 para muestra)
dev_matrix = np.array(epoch_dev_accs)
test_matrix = np.array(epoch_test_accs)
mean_dev_per_epoch = np.nanmean(dev_matrix, axis=0)
std_dev_per_epoch = np.nanstd(dev_matrix, axis=0)
mean_test_per_epoch = np.nanmean(test_matrix, axis=0)
std_test_per_epoch = np.nanstd(test_matrix, axis=0)

# Guardar estadísticas a archivo
with open(os.path.join(output_dir, file_datos), "w", encoding="utf-8") as out:
    out.write("== Estadísticas Finales ==\n")
    out.write(f"Mean Final Test Accuracy: {mean(final_accuracies):.4f}\n")
    out.write(f"Std Final Test Accuracy: {stdev(final_accuracies):.4f}\n")
    out.write(f"Mean Execution Time (s): {mean(execution_times):.2f}\n")
    out.write(f"Std Execution Time (s): {stdev(execution_times):.2f}\n\n")

    out.write("== Evolución por época ==\n")
    out.write(f"{'Epoch':<6} {'Mean Dev Acc':<15} {'Std Dev Acc':<15} {'Count Dev':<10} {'Mean Test Acc':<15} {'Std Test Acc':<15} {'Count Test':<10}\n")
    for i in range(max_len):
        # Contamos cuántos valores válidos (no-NaN) hay en cada época
        dev_count = np.count_nonzero(~np.isnan(dev_matrix[:, i]))
        test_count = np.count_nonzero(~np.isnan(test_matrix[:, i]))

        out.write(f"{i:<6} "
                  f"{mean_dev_per_epoch[i]:<15.4f} {std_dev_per_epoch[i]:<15.4f} {dev_count:<10} "
                  f"{mean_test_per_epoch[i]:<15.4f} {std_test_per_epoch[i]:<15.4f} {test_count:<10}\n")


# Gráfica Dev
plt.figure()
plt.plot(mean_dev_per_epoch, label="Dev Accuracy")
plt.fill_between(range(len(mean_dev_per_epoch)),
                 mean_dev_per_epoch - std_dev_per_epoch,
                 mean_dev_per_epoch + std_dev_per_epoch,
                 alpha=0.3)
plt.title("Evolución de la Accuracy en Dev por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, file_dev))

# Gráfica Test
plt.figure()
plt.plot(mean_test_per_epoch, label="Test Accuracy", color="orange")
plt.fill_between(range(len(mean_test_per_epoch)),
                 mean_test_per_epoch - std_test_per_epoch,
                 mean_test_per_epoch + std_test_per_epoch,
                 alpha=0.3, color="orange")
plt.title("Evolución de la Accuracy en Test por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, file_test))
