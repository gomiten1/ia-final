import pandas as pd


csv_path = 'WDBCOriginal.csv'  

# 2. Carga el CSV en un DataFrame
df = pd.read_csv(csv_path)

# 3. Muestra las primeras filas para verificar columnas y datos
print(df.head())

# 4. Checa info general (tipos, nulos, etc.)
print(df.info())

# 5. Estadísticos descriptivos básicos
print(df.describe())

# 6. Cuenta cuántos benignos vs malignos
print(df['Diagnosis'].value_counts())

