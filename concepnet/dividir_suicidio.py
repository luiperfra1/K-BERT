import pandas as pd
import os

def generar_parte_equilibrada(input_path, output_path, tamaño_total):
    df = pd.read_csv(input_path, sep='\t')

    # Convertir a string si no lo está
    df['label'] = df['label'].astype(str)

    # Separar por clase
    grupos = [df[df['label'] == label].reset_index(drop=True) for label in df['label'].unique()]

    if len(grupos) != 2:
        raise ValueError("El dataset debe tener exactamente dos clases (por ejemplo, '0' y '1').")

    tamaño_por_clase = tamaño_total // 2

    if any(len(grupo) < tamaño_por_clase for grupo in grupos):
        raise ValueError("No hay suficientes muestras para generar una parte equilibrada con ese tamaño.")

    # Tomar muestras equilibradas
    parte = [grupo.sample(n=tamaño_por_clase, random_state=42) for grupo in grupos]

    df_parte = pd.concat(parte).sample(frac=1, random_state=42)  # Mezclar

    # Guardar el nuevo archivo
    df_parte.to_csv(output_path, sep='\t', index=False)

    print(f"✅ Archivo generado: {output_path} con {len(df_parte)} muestras equilibradas.")

    # Mostrar proporción de clases
    conteo = df_parte['label'].value_counts(normalize=True) * 100
    for label, porcentaje in conteo.items():
        print(f"🟩 Clase {label}: {porcentaje:.2f}%")

# Ejemplo de uso
generar_parte_equilibrada("train.tsv", "train_small.tsv", 1000)
