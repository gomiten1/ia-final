import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


# ------------------- ADAMW OPTIMIZER -------------------
class AdamW:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-4, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k in self.params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.params[k])

# ------------------- MLP CLASS -------------------
class MLP:
    def __init__(self, layer_sizes, activations, lr=0.001, epochs=2000, seed=None, dropout_rate=0.2, patience=100):
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 2
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.patience = patience
        self._init_weights(seed)

    def _init_weights(self, seed):
        if seed is not None:
            np.random.seed(seed)
        self.weights = []
        self.biases = []
        self.dropout_masks = []
        for i in range(len(self.layer_sizes) - 1):
            in_size, out_size = self.layer_sizes[i], self.layer_sizes[i + 1]
            limit = np.sqrt(2. / in_size) if i < len(self.activations) and self.activations[i] == 'relu' else np.sqrt(1. / in_size)
            self.weights.append(np.random.randn(in_size, out_size) * limit)
            self.biases.append(np.zeros((1, out_size)))

        self.params = {f"W{i}": self.weights[i] for i in range(len(self.weights))}
        self.params.update({f"b{i}": self.biases[i] for i in range(len(self.biases))})
        self.optimizer = AdamW(self.params, lr=self.lr)

    def _activate(self, X, func):
        return 1 / (1 + np.exp(-X)) if func == 'sigmoid' else np.maximum(0, X)

    def _activate_derivative(self, X, func):
        return (X > 0).astype(float) if func == 'relu' else (1 / (1 + np.exp(-X))) * (1 - 1 / (1 + np.exp(-X)))

    def _softmax(self, X):
        e_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e_X / np.sum(e_X, axis=1, keepdims=True)

    def _forward(self, X, train=True):
        A = X
        Zs, As, masks = [], [A], []
        for i, func in enumerate(self.activations):
            Z = A @ self.weights[i] + self.biases[i]
            A = self._activate(Z, func)
            if train and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=A.shape) / (1 - self.dropout_rate)
                A *= mask
                masks.append(mask)
            else:
                masks.append(np.ones_like(A))
            Zs.append(Z)
            As.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = self._softmax(Z)
        Zs.append(Z)
        As.append(A)
        return Zs, As, masks

    def _compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        eps = 1e-15
        return -np.sum(Y_true * np.log(np.clip(Y_pred, eps, 1 - eps))) / m

    def _backward(self, Zs, As, Y_true, masks):
        grads = {}
        m = Y_true.shape[0]
        dZ = As[-1] - Y_true
        grads[f"W{len(self.weights) - 1}"] = As[-2].T @ dZ / m
        grads[f"b{len(self.biases) - 1}"] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in reversed(range(len(self.activations))):
            dA = dZ @ self.weights[i + 1].T
            dA *= masks[i]
            dZ = dA * self._activate_derivative(Zs[i], self.activations[i])
            grads[f"W{i}"] = As[i].T @ dZ / m
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
        return grads

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=16, verbose=True):
        n = X.shape[0]
        best_loss = np.inf
        patience_counter = 0
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_shuff, Y_shuff = X[idx], Y[idx]
            for i in range(0, n, batch_size):
                X_batch, Y_batch = X_shuff[i:i+batch_size], Y_shuff[i:i+batch_size]
                Zs, As, masks = self._forward(X_batch, train=True)
                grads = self._backward(Zs, As, Y_batch, masks)
                self.optimizer.step(grads)

            if verbose and (epoch + 1) % 100 == 0:
                _, As_train, _ = self._forward(X, train=False)
                train_loss = self._compute_loss(As_train[-1], Y)
                #print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}")

            if X_val is not None and Y_val is not None:
                _, As_val, _ = self._forward(X_val, train=False)
                val_loss = self._compute_loss(As_val[-1], Y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1} with val loss {val_loss:.4f}")
                        break

        self.weights = best_weights
        self.biases = best_biases

    def predict_proba(self, X):
        _, As, _ = self._forward(X, train=False)
        return As[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ------------------- CARGA DE DATOS -------------------

df = pd.read_csv("data/TCGA_InfoWithGrade.csv")

# Separar X e y
X = df.drop(columns=["Grade","Race"])  # variable objetivo
y = df["Grade"]

# Normalizar solo la columna 'Age'
if 'Age' in X.columns:
    scaler = StandardScaler()
    X['Age'] = scaler.fit_transform(X[['Age']])

# Codificar etiquetas (One-Hot)
y_encoded = pd.get_dummies(y).values  # One-hot como array NumPy

# ------------------- ENTRENAMIENTO -------------------

# Dividir en entrenamiento, validación y prueba
X_temp, X_test, y_temp, y_test = train_test_split(X.values, y_encoded, test_size=0.2, random_state=698)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=698)

# Estructura de la red: input -> hidden layers -> output
input_size = X_train.shape[1]
output_size = y_train.shape[1]

mlp = MLP(
    layer_sizes=[input_size, 64, 32, output_size],
    activations=['relu',  'relu'],
    lr=0.01,
    epochs=2000,
    dropout_rate=0.3,
    patience=100,
    seed=698
)

# Entrenar con validación (early stopping activado)
mlp.fit(X_train, y_train, X_val=X_val, Y_val=y_val, batch_size=18, verbose=True)

menu = int(input("¿Quieres ver los resultados de la evaluación del modelo? (1: Sí, 0: No): "))
if menu == 1:
    # --- Predicciones ---
    y_pred = mlp.predict(X_test)
    y_true = np.argmax(y_test, axis=1)

    # --- Accuracy y métricas ---
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # --- Factor de certeza ---
    def attach_certainty(prob_vector, classes):
        class_idx = np.argmax(prob_vector)
        return classes[class_idx], prob_vector[class_idx]

    y_prob = mlp.predict_proba(X_test)  
    results = []
    for probs in y_prob:
        cls, cf = attach_certainty(probs, classes=[0, 1])
        results.append({'Predicted': cls, 'Certainty': cf})

    results_df = pd.DataFrame(results)
    print("\nEjemplos con factor de certeza:\n", results_df.head())

    # --- Curva ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])  # Asumiendo clase positiva es 1
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - MLP')
    plt.legend(loc="lower right")
    plt.show()

menu = int(input("¿Quieres realizar una consulta? (1: Sí, 0: No): "))
if menu == 1:
    print("\nIngresa los valores de las características para predecir el grado del tumor:")

    # Entrada manual
    gender = int(input("Gender (0: Hombre, 1: Mujer): "))
    age = float(input("Age at diagnosis (ej. 45.3): "))

    # Genes binarios
    genes = ['IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1',
             'NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']

    gene_values = []
    for gene in genes:
        val = int(input(f"{gene} (0: no mutado, 1: mutado): "))
        gene_values.append(val)

    # Crear DataFrame con los datos
    input_data = [gender, age] + gene_values
    columns = ['Gender', 'Age_at_diagnosis'] + genes
    df_new = pd.DataFrame([input_data], columns=columns)

    # Selección de características en el orden usado por el modelo
    features_for_model = ['Gender', 'Age_at_diagnosis'] + genes
    X_new = df_new[features_for_model].astype(float).values

    # Predicción
    pred = mlp.predict(X_new)[0]
    prob = mlp.predict_proba(X_new)[0]
    predicted_class = pred
    certainty = np.max(prob)

    label_map = {0: "LGG (Low Grade Glioma)", 1: "GBM (Glioblastoma)"}

    print(f"\nPredicción: Clase  {pred} → {label_map[pred]}")
