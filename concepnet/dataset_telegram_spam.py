import pandas as pd
from sklearn.utils import shuffle
import os
import re
import emoji

# Eliminar solo emojis visuales (no emoticonos de texto)
def eliminar_emoticonos(texto):
    return emoji.replace_emoji(texto, replace='')

# Limpieza completa del texto
def limpiar_texto(texto):
    texto = texto.replace("\n", " ").replace("\r", " ").strip()
    texto = eliminar_emoticonos(texto)
    texto = re.sub(r"\x01", "", texto)               # elimina carácter extraño
    texto = re.sub(r"\b1635465\b", "", texto)        # elimina el número basura
    texto = re.sub(r"\s+", " ", texto).strip()       # normaliza espacios
    return texto

# Cargar el CSV
df = pd.read_csv("telegram_spam.csv", on_bad_lines='skip')

# Renombrar columnas si fuera necesario
if "Email Text" in df.columns and "Email Type" in df.columns:
    df = df.rename(columns={"Email Text": "text_a", "Email Type": "label"})
elif "text" in df.columns and "label" in df.columns:
    df = df.rename(columns={"text": "text_a", "label": "label"})
else:
    df.columns = ["label", "text_a"]  # fallback

# Eliminar textos vacíos
df = df[df["text_a"].astype(str).str.strip() != ""]

# Limpiar cada texto
df["text_a"] = df["text_a"].astype(str).apply(limpiar_texto)

# Convertir etiquetas: "spam" → "1", "ham" → "0"
df["label"] = df["label"].replace({"spam": "1", "ham": "0"})

# Separar por clase
grupo_0 = df[df["label"] == "0"]
grupo_1 = df[df["label"] == "1"]

# Tamaño mínimo para balancear (máximo 1500 por clase)
tamaño_minimo = min(len(grupo_0), len(grupo_1), 1500)

# Tomar muestras balanceadas
grupo_0 = grupo_0.sample(n=tamaño_minimo, random_state=42)
grupo_1 = grupo_1.sample(n=tamaño_minimo, random_state=42)

# Dividir en tercios
def dividir_grupo(grupo):
    tercio = len(grupo) // 3
    return grupo.iloc[:tercio], grupo.iloc[tercio:2*tercio], grupo.iloc[2*tercio:]

grupo_0_train, grupo_0_dev, grupo_0_test = dividir_grupo(grupo_0)
grupo_1_train, grupo_1_dev, grupo_1_test = dividir_grupo(grupo_1)

# Combinar y mezclar
train_df = shuffle(pd.concat([grupo_0_train, grupo_1_train]), random_state=42).reset_index(drop=True)
dev_df = shuffle(pd.concat([grupo_0_dev, grupo_1_dev]), random_state=42).reset_index(drop=True)
test_df = shuffle(pd.concat([grupo_0_test, grupo_1_test]), random_state=42).reset_index(drop=True)

# Guardar TSV
os.makedirs("telegram", exist_ok=True)
train_df.to_csv("telegram/train.tsv", sep="\t", index=False)
dev_df.to_csv("telegram/dev.tsv", sep="\t", index=False)
test_df.to_csv("telegram/test.tsv", sep="\t", index=False)

# Imprimir distribución de clases
def print_dist(df, name):
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"\n{name}:")
    for label, pct in dist.items():
        print(f"  Clase {label}: {pct:.2f}%")

print_dist(train_df, "Train")
print_dist(dev_df, "Dev")
print_dist(test_df, "Test")
