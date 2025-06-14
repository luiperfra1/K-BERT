import pandas as pd
from sklearn.utils import shuffle
import os

# Cargar y limpiar el CSV
df = pd.read_csv("Toxicity_detection_context.csv", on_bad_lines='skip')
df = df.rename(columns={"text": "text_a", " label": "label"})
df = df[["label", "text_a"]]
df = df[df["text_a"].astype(str).str.strip() != ""]
df["text_a"] = df["text_a"].astype(str).str.replace(r"[\r\n]+", " ", regex=True)
df["label"] = df["label"].astype(str)

# Separar por clases
grupo_0 = df[df["label"] == "0"]
grupo_1 = df[df["label"] == "1"]

# Comprobar que hay suficientes muestras
if len(grupo_0) < 1500 or len(grupo_1) < 1500:
    print(f"Grupo 0: {len(grupo_0)} muestras, Grupo 1: {len(grupo_1)} muestras")
    raise ValueError("No hay suficientes muestras de ambas clases (se necesitan al menos 1500 por clase).")

# Tomar 1500 muestras de cada clase
grupo_0 = grupo_0.sample(n=1500, random_state=42)
grupo_1 = grupo_1.sample(n=1500, random_state=42)

# Dividir cada grupo en 3 bloques de 500
grupo_0_train = grupo_0.iloc[:500]
grupo_0_dev = grupo_0.iloc[500:1000]
grupo_0_test = grupo_0.iloc[1000:1500]

grupo_1_train = grupo_1.iloc[:500]
grupo_1_dev = grupo_1.iloc[500:1000]
grupo_1_test = grupo_1.iloc[1000:1500]

# Combinar y mezclar
train_df = shuffle(pd.concat([grupo_0_train, grupo_1_train]), random_state=42).reset_index(drop=True)
dev_df = shuffle(pd.concat([grupo_0_dev, grupo_1_dev]), random_state=42).reset_index(drop=True)
test_df = shuffle(pd.concat([grupo_0_test, grupo_1_test]), random_state=42).reset_index(drop=True)

# Crear carpeta y guardar
os.makedirs("toxic_balanced", exist_ok=True)
train_df.to_csv("toxic_balanced/train.tsv", sep="\t", index=False)
dev_df.to_csv("toxic_balanced/dev.tsv", sep="\t", index=False)
test_df.to_csv("toxic_balanced/test.tsv", sep="\t", index=False)

# Imprimir distribución
def print_dist(df, name):
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"\n{name}:")
    for label, pct in dist.items():
        print(f"  Clase {label}: {pct:.2f}%")

print_dist(train_df, "Train")
print_dist(dev_df, "Dev")
print_dist(test_df, "Test")
