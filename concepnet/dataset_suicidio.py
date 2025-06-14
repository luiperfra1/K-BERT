import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar CSV original
df = pd.read_csv("twitter-suicidal_data.csv")

# Renombrar columnas
df = df.rename(columns={"tweet": "text_a", "intention": "label"})

# Reordenar columnas
df = df[["label", "text_a"]]

# Eliminar filas con text_a vacío o solo espacios
df = df[df["text_a"].astype(str).str.strip() != ""]

# Asegurar que label es str o int para stratify
df["label"] = df["label"].astype(str)

# Dividir en train (70%) y resto (30%) con balance de clases
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

# Dividir el 30% restante en dev y test (15% cada uno)
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

# Guardar los tres archivos como TSV
train_df.to_csv("train.tsv", sep="\t", index=False)
dev_df.to_csv("dev.tsv", sep="\t", index=False)
test_df.to_csv("test.tsv", sep="\t", index=False)
