import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class GliomaClassifierAgent:
    """
    Agente clasificador de gliomas basado en árbol de decisión.
    Permite obtener certeza de cada predicción.
    """
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
        self.trained = False

    def fit(self, X, y):
        """Entrena el modelo con datos de entrenamiento."""
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        """
        Realiza predicciones y devuelve etiquetas junto con factor de certeza.
        CF = 2 * P(predicho) - 1  # Rango [-1,1]
        """
        if not self.trained:
            raise RuntimeError("El modelo no ha sido entrenado.")
        probs = self.model.predict_proba(X)
        labels = self.model.predict(X)
        # Obtener probabilidad de la clase predicha
        cf = [2 * probs[i, lab] - 1 for i, lab in enumerate(labels)]
        return labels, cf

    def evaluate(self, X_test, y_test):
        """Mide desempeño usando métricas estándar."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['LGG', 'GBM']
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'roc_curve': (fpr, tpr)
        }

    def export_tree(self, feature_names):
        """Exporta la representación textual del árbol."""
        return export_text(self.model, feature_names=feature_names)
    
def get_user_input(feature_names):
    """
    Solicita al usuario que ingrese valores para cada feature.
    Retorna un DataFrame de una sola fila.
    """
    
    race_map = {
        'white': 0,
        'black': 1,
        'african american': 1,
        'asian': 2,
        'american indian': 3,
        'alaska native': 3
    }
    gender_map = {'male': 0, 'female': 1}
    user_data = {}
    print("\nIngrese los siguientes valores:")
    for feat in feature_names:
        if feat == 'Age_at_diagnosis':
            while True:
                val = input(f" - {feat} (edad en años, puede incluir decimal): ")
                try:
                    user_data[feat] = float(val)
                    break
                except ValueError:
                    print("Valor inválido, inténtalo de nuevo.")
        elif feat == 'Gender':
            while True:
                val = input(f"- {feat} (male, female): ").strip().lower()
                if val in gender_map:
                    user_data[feat] = gender_map[val]
                    break
                else:
                    print("Ingresa 'male' o 'female'.")
        elif feat == 'Race':
            while True:
                val = input(f"- {feat} (white, black, african american, asian, american indian, alaska native): ").strip().lower()
                if val in race_map:
                    user_data[feat] = race_map[val]
                    break
                else:
                    print("Valor inválido")
        else:
            while True:
                val = input(f"¿Mutación en {feat}? (yes/no): ").strip().lower()
                if val in ('yes', 'no'):
                    user_data[feat] = 1 if val == 'yes' else 0
                    break
                else:
                    print("Ingresa 'yes' o 'no'.")
        
        
    return pd.DataFrame([user_data])

if __name__ == '__main__':
    # Cargar dataset
    df = pd.read_csv('data/TCGA_InfoWithGrade.csv')

 

    # Separar características y etiqueta
    X = df.drop(columns=['Grade', 'ID']) if 'ID' in df.columns else df.drop(columns=['Grade'])
    y = df['Grade']

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar agente
    agent = GliomaClassifierAgent(max_depth=5)
    agent.fit(X_train, y_train)
    
    menu = int(input("¿Quieres realizar una consulta? (1: Sí, 0: No): "))
    if menu == 1:
    
        # Usuario
        feature_names = list(X.columns)
        user_df = get_user_input(feature_names)
        labels, certainties = agent.predict(user_df)
        label = 'GBM' if labels[0] == 1 else 'LGG'
        cf = certainties[0]
        print(f"\nPredicción para el paciente ingresado: {label} (certeza={cf:.2f})")

    
    menu = int(input("¿Quieres ver los resultados de la evaluación del modelo? (1: Sí, 0: No): "))
    if menu == 1:
        # Evaluar desempeño
        results = agent.evaluate(X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.2f}")
        print("Classification Report:\n", results['classification_report'])
        print("Confusion Matrix:\n", results['confusion_matrix'])
        print(f"ROC AUC: {results['roc_auc']:.2f}")

        # Ejemplo de predicción con certeza
        sample = X_test.iloc[:5]
        labels, certainties = agent.predict(sample)
        for i, (lab, cf) in enumerate(zip(labels, certainties)):
            print(f"Muestra {i}: Predicción = {'GBM' if lab==1 else 'LGG'}, Certeza = {cf:.2f}")

        # Mostrar estructura del árbol
        print("\nÁrbol de decisión:\n")
        print(agent.export_tree(feature_names=list(X.columns)))
        
        fpr, tpr = results['roc_curve']
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.show()
