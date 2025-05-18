import pandas as pd

# Carga el CSV
df = pd.read_csv("data/tiroides.csv")  # reemplaza con el nombre real del archivo

# Lista de columnas de interés
columnas = [
    "Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy",
    "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology",
    "Focality", "Risk", "T", "N", "M", "Stage", "Response", "Recurred"
]

# Itera sobre cada columna y muestra sus valores únicos
for col in columnas:
    if col in df.columns:
        valores_unicos = df[col].dropna().unique()
        print(f"\n{col} ({len(valores_unicos)} valores únicos):")
        print(sorted(valores_unicos))
    else:
        print(f"\n{col}: No se encontró en el CSV.")
