import os
import pandas as pd
import matplotlib.pyplot as plt

# Rutas
base_dir = "estadisticas/toxicidad"
file_sinKG = os.path.join(base_dir, "sinKG", "toxicidadsinKG_datos.txt")
file_conKG = os.path.join(base_dir, "conKG", "toxicidadconKG_datos.txt")
dev= "toxicidad_dev_evolution.png"
test= "toxicidad_test_evolution.png"

# Función para extraer la tabla de evolución del .txt
def cargar_tabla_evolucion(path_txt):
    with open(path_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if line.startswith("== Evolución por época ==")) + 2
    data = [line.strip().split() for line in lines[start_idx:] if line.strip()]
    df = pd.DataFrame(data, columns=["Epoch", "Mean_Dev", "Std_Dev", "Mean_Test", "Std_Test"])
    df = df.apply(pd.to_numeric)
    return df

# Cargar datos
df_sinKG = cargar_tabla_evolucion(file_sinKG)
df_conKG = cargar_tabla_evolucion(file_conKG)

# Función para plotear evolución comparativa
def plot_evolucion_comparada(df1, df2, col_mean, col_std, label1, label2, output_path, title, color1="blue", color2="green"):
    plt.figure(figsize=(10, 5))
    epochs = df1["Epoch"]

    plt.plot(epochs, df1[col_mean], label=label1, color=color1)
    plt.fill_between(epochs,
                     df1[col_mean] - df1[col_std],
                     df1[col_mean] + df1[col_std],
                     alpha=0.2, color=color1)

    plt.plot(epochs, df2[col_mean], label=label2, color=color2)
    plt.fill_between(epochs,
                     df2[col_mean] - df2[col_std],
                     df2[col_mean] + df2[col_std],
                     alpha=0.2, color=color2)

    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Guardado: {output_path}")

# Crear gráficas combinadas
plot_evolucion_comparada(
    df_sinKG, df_conKG,
    "Mean_Dev", "Std_Dev",
    "Sin KG", "Con KG",
    os.path.join(base_dir, dev),
    "Evolución de Accuracy en Dev (Sin KG vs Con KG)"
)

plot_evolucion_comparada(
    df_sinKG, df_conKG,
    "Mean_Test", "Std_Test",
    "Sin KG", "Con KG",
    os.path.join(base_dir, test),
    "Evolución de Accuracy en Test (Sin KG vs Con KG)"
)
