import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Cargar el CSV original
df = pd.read_csv("text_classification.csv")

# Renombrar columnas
df = df.rename(columns={"Text": "text_a", "Label": "label"})

# Reordenar columnas
df = df[["label", "text_a"]]

# Eliminar filas con text_a vacío o solo espacios
df = df[df["text_a"].astype(str).str.strip() != ""]

# Limpiar saltos de línea en el texto (convertirlos a espacios)
df["text_a"] = df["text_a"].astype(str).str.replace(r"[\r\n]+", " ", regex=True)

# Asegurar que label sea string (por si acaso)
df["label"] = df["label"].astype(str)

# Dividir el dataset en 1/3 para train y 2/3 restante
train_df, temp_df = train_test_split(
    df,
    test_size=2/3,
    stratify=df["label"],
    random_state=42
)

# Dividir el 2/3 restante entre dev y test (cada uno 1/3 del total)
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

# Crear carpeta de salida si no existe
os.makedirs("text_class", exist_ok=True)

# Guardar los tres archivos como TSV
train_df.to_csv("text_class/train.tsv", sep="\t", index=False)
dev_df.to_csv("text_class/dev.tsv", sep="\t", index=False)
test_df.to_csv("text_class/test.tsv", sep="\t", index=False)

# Imprimir proporción de clases
def print_class_distribution(df, name):
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"\nDistribución de clases en {name}:")
    for label, pct in dist.items():
        print(f"  Clase {label}: {pct:.2f}%")

print_class_distribution(train_df, "train")
print_class_distribution(dev_df, "dev")
print_class_distribution(test_df, "test")
