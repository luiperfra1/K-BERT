import pandas as pd
from sklearn.utils import shuffle
import os
import re

# Cargar el CSV
df = pd.read_csv("news_class.csv", on_bad_lines='skip')

# Renombrar columnas
df = df.rename(columns={"News": "text_a", "Category": "label"})

# Eliminar textos vacíos
df = df[df["text_a"].astype(str).str.strip() != ""]

# Limpiar saltos de línea, corchetes numéricos y espacios innecesarios
df["text_a"] = df["text_a"].astype(str)
df["text_a"] = df["text_a"].str.replace(r"[\r\n]+", " ", regex=True)
df["text_a"] = df["text_a"].str.replace(r"\[\d+\]", "", regex=True).str.strip()

# Convertir etiquetas: "Health" → "0", "Sports" → "1"
df["label"] = df["label"].replace({"Health": "0", "Sports": "1"})

# Separar por clase
grupo_0 = df[df["label"] == "0"]  # Health
grupo_1 = df[df["label"] == "1"]  # Sports

# Determinar cuántas muestras usar (máx 1500 por clase)
tamaño_minimo = min(len(grupo_0), len(grupo_1), 1500)

# Tomar muestras balanceadas
grupo_0 = grupo_0.sample(n=tamaño_minimo, random_state=42)
grupo_1 = grupo_1.sample(n=tamaño_minimo, random_state=42)

# Dividir cada grupo en 3 partes iguales
def dividir_grupo(grupo):
    tercio = len(grupo) // 3
    return grupo.iloc[:tercio], grupo.iloc[tercio:2*tercio], grupo.iloc[2*tercio:]

grupo_0_train, grupo_0_dev, grupo_0_test = dividir_grupo(grupo_0)
grupo_1_train, grupo_1_dev, grupo_1_test = dividir_grupo(grupo_1)

# Combinar y mezclar
train_df = shuffle(pd.concat([grupo_0_train, grupo_1_train]), random_state=42).reset_index(drop=True)
dev_df = shuffle(pd.concat([grupo_0_dev, grupo_1_dev]), random_state=42).reset_index(drop=True)
test_df = shuffle(pd.concat([grupo_0_test, grupo_1_test]), random_state=42).reset_index(drop=True)

# Guardar archivos TSV
output_dir = "news_balanced"
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(f"{output_dir}/train.tsv", sep="\t", index=False)
dev_df.to_csv(f"{output_dir}/dev.tsv", sep="\t", index=False)
test_df.to_csv(f"{output_dir}/test.tsv", sep="\t", index=False)

# Mostrar distribución por consola
def print_dist(df, name):
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"\n{name}:")
    for label, pct in dist.items():
        print(f"  Clase {label}: {pct:.2f}%")

print_dist(train_df, "Train")
print_dist(dev_df, "Dev")
print_dist(test_df, "Test")
