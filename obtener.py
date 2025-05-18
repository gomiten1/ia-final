import pandas as pd

df = pd.read_csv("data/tiroides.csv")  


columnas = [
    "Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy",
    "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology",
    "Focality", "Risk", "T", "N", "M", "Stage", "Response", "Recurred"
]


for col in columnas:
    if col in df.columns:
        valores_unicos = df[col].dropna().unique()
        print(f"\n{col} ({len(valores_unicos)} valores únicos):")
        print(sorted(valores_unicos))
    else:
        print(f"\n{col}: No se encontró en el CSV.")
