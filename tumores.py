

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils import resample
import matplotlib.pyplot as plt

# 1. Cargar y preprocesar datos
df = pd.read_csv('data/tumores.csv')
# Eliminar columna de ID
df = df.drop(columns=['IDNumber'])
# Mapear diagnóstico a numérico: M -> 1, B -> 0
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# 2. Balancear el dataset con submuestreo (downsampling)
# Separar clases
df_maligno = df[df['Diagnosis'] == 1]
df_benigno = df[df['Diagnosis'] == 0]

# Submuestrear la clase mayoritaria (benignos) al tamaño de malignos
df_benigno_down = resample(
    df_benigno,
    replace=False,
    n_samples=len(df_maligno),
    random_state=42
)

# Combinar el dataset balanceado
df_bal = pd.concat([df_maligno, df_benigno_down])

# 3. Separar características y etiqueta
a = df_bal.drop(columns=['Diagnosis'])
y = df_bal['Diagnosis']

# 4. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    a, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 5. Entrenar modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar desempeño
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva (maligno)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_proba)
}

print("--- Métricas de evaluación ---")
for name, val in metrics.items():
    print(f"{name}: {val:.3f}")

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 7. Razonamiento con incertidumbre
def factor_de_certeza(proba, umbral=0.5):
    """
    Convierte una probabilidad en un factor de certeza (-1 a 1):
    CF = (P_clase - umbral) / (1 - umbral) si proba >= umbral,
    CF = (proba - umbral) / umbral si proba < umbral.
    """
    if proba >= umbral:
        return (proba - umbral) / (1 - umbral)
    return (proba - umbral) / umbral

# Aplicar al conjunto de prueba
test_results = X_test.copy()
test_results['y_true'] = y_test
test_results['y_pred'] = y_pred
test_results['proba_maligno'] = y_proba
test_results['CF'] = test_results['proba_maligno'].apply(lambda p: factor_de_certeza(p, umbral=0.5))

print("\nAlgunas predicciones con factor de certeza:")
print(test_results[['y_true', 'y_pred', 'proba_maligno', 'CF']].head())

# 8. Graficar curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['ROC AUC']:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Árbol de Decisión')
plt.legend()
plt.show()

#  9. Usuario
print("\nIngresa las medidas del tumor para clasificar:")
feature_names = [
    'Radius', 'Texture', 'Perimeter', 'Area',
    'Smoothness', 'Compactness', 'Concavity',
    'ConcavePoints', 'Symmetry', 'FractalDimension'
]
user_vals = []
for feat in feature_names:
    while True:
        try:
            val = float(input(f"{feat}: "))
            user_vals.append(val)
            break
        except ValueError:
            print("Por favor ingresa un número válido.")

# Crear DataFrame de entrada
df_user = pd.DataFrame([user_vals], columns=feature_names)

# Predecir y calcular factor de certeza
user_pred = model.predict(df_user)[0]
user_proba = model.predict_proba(df_user)[0, 1]
user_cf = factor_de_certeza(user_proba)

resultado = 'Maligno' if user_pred == 1 else 'Benigno'
print(f"\nResultado: {resultado}\nProbabilidad maligno: {user_proba:.2f}\nFactor de certeza: {user_cf:.2f}")
