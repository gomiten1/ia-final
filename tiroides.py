import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
# 1. Cargar el dataset
df = pd.read_csv('data/tiroides.csv')

# 2. Definir variables predictoras y objetivo
y = df['Recurred']  # objetivo sí/no
X = df.drop(columns=['Recurred'])

# 3. Preprocesamiento de variables categóricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Combinar con la columna Age
df_num = X[['Age']].to_numpy()
X_preprocessed = np.hstack([df_num, X_encoded])
feature_names = ['Age'] + encoded_cols.tolist()

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluar desempeño


# 7. Razonamiento con incertidumbre
def attach_certainty(proba_array, classes):
    idx = np.argmax(proba_array)
    return classes[idx], proba_array[idx]


menu = int(input("¿Quieres realizar una consulta? (1: Sí, 0: No): "))
if menu == 1:
# 9. Sección interactiva para nuevas mediciones
    print("\nIngresa nuevos valores para predecir recurrencia:")
    # Recopilar inputs del usuario
    input_age = int(input("Age: "))
    input_values = [input_age]
    for col in categorical_cols:
        val = input(f"{col} ({encoder.categories_[categorical_cols.index(col)]}): ")
        input_values.append(val)

    # Convertir entrada a DataFrame
    df_new = pd.DataFrame([input_values], columns=['Age'] + categorical_cols)
    # Codificar categóricas
    X_new_encoded = encoder.transform(df_new[categorical_cols])
    X_new = np.hstack([np.array(df_new[['Age']]), X_new_encoded])

    # Predecir
    pred = clf.predict(X_new)[0]
    prob = clf.predict_proba(X_new)[0]
    # Obtener certeza
    pred_class, certainty = attach_certainty(prob, clf.classes_)
    print(f"\nPredicción: {pred_class} con certeza de {certainty:.2f}")


# Presentar resultados
menu = int(input("¿Quieres ver los resultados de la evaluación del modelo? (1: Sí, 0: No): "))
if menu == 1:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results = []
    for probs in y_prob:
        cls, cf = attach_certainty(probs, clf.classes_)
        results.append({'Predicted': cls, 'Certainty': cf})

    results_df = pd.DataFrame(results)
    print("\nEjemplos con factor de certeza:\n", results_df.head())

    # 8. Visualizar la estructura del árbol (opcional)
    text_tree = export_text(clf, feature_names=feature_names)
    print("\nÁrbol de decisión:\n", text_tree)
    
    # ROC
    y_test_bin = (y_test == 'Yes').astype(int)
    y_prob_pos = y_prob[:, list(clf.classes_).index('Yes')]
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
