import pandas as pd
from sklearn.utils import shuffle
import os

# Cargar el CSV
df = pd.read_csv("Phishing_Email.csv", on_bad_lines='skip')

# Renombrar columnas
df = df.rename(columns={"Email Text": "text_a", "Email Type": "label"})

# Eliminar columna innecesaria si está presente
df = df[["label", "text_a"]]

# Eliminar textos vacíos
df = df[df["text_a"].astype(str).str.strip() != ""]

# Limpiar saltos de línea y espacios innecesarios
df["text_a"] = df["text_a"].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()

# Convertir etiquetas: "Phishing Email" → "1", "Safe Email" → "0"
df["label"] = df["label"].replace({"Phishing Email": "1", "Safe Email": "0"})

# Separar por clase
grupo_0 = df[df["label"] == "0"]
grupo_1 = df[df["label"] == "1"]

# Determinar cuántas muestras usar (balanceado)
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
os.makedirs("phishing_balanced", exist_ok=True)
train_df.to_csv("phishing_balanced/train.tsv", sep="\t", index=False)
dev_df.to_csv("phishing_balanced/dev.tsv", sep="\t", index=False)
test_df.to_csv("phishing_balanced/test.tsv", sep="\t", index=False)

# Mostrar distribución por consola
def print_dist(df, name):
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"\n{name}:")
    for label, pct in dist.items():
        print(f"  Clase {label}: {pct:.2f}%")

print_dist(train_df, "Train")
print_dist(dev_df, "Dev")
print_dist(test_df, "Test")
