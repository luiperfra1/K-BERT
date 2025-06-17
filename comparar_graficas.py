import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Rutas
base_dir = "estadisticas/salud_mental"
file_sinKG = os.path.join(base_dir, "sinKG", "salud_mentalsinKG_datos.txt")
file_conKG = os.path.join(base_dir, "conKG", "salud_mentalconKG_datos.txt")
dev = "salud_mental_dev_evolution.png"
test = "salud_mental_test_evolution.png"

# Función para extraer la tabla de evolución del .txt
def cargar_tabla_evolucion(path_txt):
    with open(path_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if line.startswith("== Evolución por época ==")) + 2
    data = [line.strip().split() for line in lines[start_idx:] if line.strip()]
    df = pd.DataFrame(data, columns=[
        "Epoch", "Mean_Dev", "Std_Dev", "Count_Dev", "Mean_Test", "Std_Test", "Count_Test"
    ])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# Cargar datos
df_sinKG = cargar_tabla_evolucion(file_sinKG)
df_conKG = cargar_tabla_evolucion(file_conKG)

# Función de graficado
def plot_evolucion_comparada(df1, df2, col_mean, col_std, label1, label2, output_path, title, color1="blue", color2="green"):
    plt.figure(figsize=(12, 6))
    epochs = df1["Epoch"]

    count_col1 = "Count_Dev" if "Dev" in col_mean else "Count_Test"
    count_col2 = "Count_Dev" if "Dev" in col_mean else "Count_Test"
    counts1 = df1[count_col1]
    counts2 = df2[count_col2]

    mask1 = (counts1 > 0) & (~df1[col_mean].isna())
    mask2 = (counts2 > 0) & (~df2[col_mean].isna())

    # Curvas y bandas de desviación
    plt.plot(df1["Epoch"][mask1], df1[col_mean][mask1], label=label1, color=color1)
    plt.fill_between(df1["Epoch"],
                     np.where(mask1, df1[col_mean] - df1[col_std], np.nan),
                     np.where(mask1, df1[col_mean] + df1[col_std], np.nan),
                     alpha=0.2, color=color1)

    plt.plot(df2["Epoch"][mask2], df2[col_mean][mask2], label=label2, color=color2)
    plt.fill_between(df2["Epoch"],
                     np.where(mask2, df2[col_mean] - df2[col_std], np.nan),
                     np.where(mask2, df2[col_mean] + df2[col_std], np.nan),
                     alpha=0.2, color=color2)

    # Personalizar las etiquetas del eje X
    xtick_labels = [
        f"{int(epoch)}\n{int(counts1[i])}({label1})/{int(counts2[i])}({label2})"
        for i, epoch in enumerate(epochs)
    ]
    plt.xticks(ticks=epochs, labels=xtick_labels)

    # Ejes y estilo
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Guardado: {output_path}")

# Crear gráficas comparadas
plot_evolucion_comparada(
    df_sinKG, df_conKG,
    "Mean_Dev", "Std_Dev",
    "SinKG", "ConKG",
    os.path.join(base_dir, dev),
    "Evolución de Accuracy en Dev (Sin KG vs Con KG)"
)

plot_evolucion_comparada(
    df_sinKG, df_conKG,
    "Mean_Test", "Std_Test",
    "SinKG", "ConKG",
    os.path.join(base_dir, test),
    "Evolución de Accuracy en Test (Sin KG vs Con KG)"
)
