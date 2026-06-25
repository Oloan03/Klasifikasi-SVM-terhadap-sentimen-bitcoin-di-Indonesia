import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# 1. Baca dataset
df = pd.read_csv('../labeling/3_labeled_data.csv')
X = df['stemming'].fillna('')
y = df['senti'].map({'positive': 1, 'negative': 0})

# 2. Stratified split (80:20)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 3. TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=None)

# 4. Parameter grid
param_grids = {
    'linear': {'C': [0.1, 1, 10, 100]},
    'rbf': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]},
    'poly': {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}
}

# 5. Simpan hasil
results = []

def evaluate_model(name, y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    cm = confusion_matrix(y_true, y_pred)
    results.append({
        'Model': name,
        'F1-Macro': f1_macro,
        'F1-Micro': f1_micro,
        'Confusion Matrix': cm
    })
    print(f"\n=== {name} ===")
    print("F1-Macro:", round(f1_macro, 4))
    print("Confusion Matrix:\n", cm)

# 6. Jalankan 9 model
for kernel in ['linear', 'rbf', 'poly']:
    print(f"\n{'='*50}\nRunning models with kernel: {kernel}\n{'='*50}")
    
    # --- Baseline ---
    pipe_base = Pipeline([('tfidf', tfidf), ('svm', SVC(kernel=kernel, random_state=42))])
    gs_base = GridSearchCV(pipe_base, {f'svm__{k}': v for k, v in param_grids[kernel].items()},
                           cv=10, scoring='f1', n_jobs=-1)
    gs_base.fit(X_train, y_train)
    pred_base = gs_base.predict(X_test)
    evaluate_model(f"{kernel}_baseline", y_test, pred_base)

    # --- class_weight='balanced' ---
    pipe_cw = Pipeline([('tfidf', tfidf), ('svm', SVC(kernel=kernel, class_weight='balanced', random_state=42))])
    gs_cw = GridSearchCV(pipe_cw, {f'svm__{k}': v for k, v in param_grids[kernel].items()},
                         cv=10, scoring='f1', n_jobs=-1)
    gs_cw.fit(X_train, y_train)
    pred_cw = gs_cw.predict(X_test)
    evaluate_model(f"{kernel}_class_weight", y_test, pred_cw)

    # --- SMOTE (hanya pada data latih) ---
    if kernel == 'poly':
        # Hindari degree tinggi + SMOTE (komputasi berat); batasi degree=2
        smote_grid = {'svm__C': [0.1, 1, 10], 'svm__degree': [2], 'svm__gamma': ['scale']}
    else:
        smote_grid = {f'svm__{k}': v for k, v in param_grids[kernel].items()}

    pipe_smote = Pipeline([
        ('tfidf', tfidf),
        ('smote', SMOTE(random_state=42)),
        ('svm', SVC(kernel=kernel, random_state=42))
    ])
    gs_smote = GridSearchCV(pipe_smote, smote_grid, cv=10, scoring='f1', n_jobs=-1)
    gs_smote.fit(X_train, y_train)
    pred_smote = gs_smote.predict(X_test)
    evaluate_model(f"{kernel}_SMOTE", y_test, pred_smote)

# 7. Tampilkan ringkasan F1-Macro
print("\n\n" + "="*60)
print("RINGKASAN F1-MACRO SEMUA MODEL")
print("="*60)
summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'Confusion Matrix'} for r in results])
print(summary.sort_values('F1-Macro', ascending=False).to_string(index=False))