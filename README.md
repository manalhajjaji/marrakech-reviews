# Analyse des Sentiments des Touristes à Marrakech  

**Projet de Module – 3ème année Cycle Ingénieur IDSCC (Ingénierie Data Science & Cloud Computing)**  
École Nationale des Sciences Appliquées d'Oujda – Année 2025-2026  


---

## 📌 Description du Projet

Ce projet consiste à développer un système complet d'**analyse de sentiments** appliqué aux avis touristiques sur la ville de **Marrakech**.  

L'objectif principal est d'évaluer la perception globale des touristes (Positive / Négative / Neutre) à partir d'avis textuels, en comparant trois approches différentes de NLP :

1. **VADER** → Modèle rule-based et lexicon-based (optimisé pour les textes expressifs et informels)  
2. **RoBERTa** → Modèle Transformer pré-entraîné et fine-tuné (approche deep learning state-of-the-art)  
3. **Modèle From Scratch** → Régression logistique multiclasse avec vectorisation TF-IDF (baseline classique)

Un **dashboard interactif** a été développé avec Streamlit pour visualiser et comparer les résultats des trois modèles.

---

## 🚀 Fonctionnalités Principales

- Exploration et prétraitement des données textuelles
- Application de trois modèles d'analyse de sentiments
- Évaluation détaillée (accuracy, precision, recall, F1-score, matrice de confusion)
- Visualisations claires (bar charts, pie charts, matrices de confusion)
- Dashboard multi-pages avec explications pédagogiques
- Déploiement public sur **Streamlit Community Cloud**

---

## 🏆 Résultats Obtenus (sur le jeu de test)

| Modèle              | Accuracy |
|---------------------|----------|
| From Scratch (Logistic Regression + TF-IDF) | **88.86%** |
| RoBERTa (Transformer)                        | 77.72%  |
| VADER (Lexicon-based)                        | 59.43%  |

**Conclusion clé** : La baseline classique (From Scratch) surpasse largement les modèles plus complexes grâce à un pré-traitement adapté et une vectorisation TF-IDF optimisée sur le corpus spécifique.

---

## 🗂️ Architecture du Projet

```
MARRAKECH-REVIEWS/
├── app/
│   ├── assets/
│   │   └── marrakech.jpg
│   ├── pages/
│   │   ├── 1_Accueil.py
│   │   ├── 02_VADER_Analysis.py
│   │   ├── 03_RoBERTa_Analysis.py
│   │   └── 04_Logistic_Regression.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── dataset_marrakech_english.csv
│   └── processed/
│       ├── results_vader.csv / results_vader_test.csv
│       ├── results_roberta.csv / results_roberta_test.csv
│       └── results_fromscratch_test.csv
│       ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
├── models/
│   ├── logreg_sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_new_generator.ipynb          → Génération synthétique des données
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_vader_analysis.ipynb
│   ├── 05_roberta_analysis.ipynb
│   └── 06_scratch_model.ipynb
├── .env
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Utilisées

- **Python**  
- **Pandas, NLTK, Scikit-learn** → Pré-traitement et modèle from scratch  
- **VADER Sentiment** → Analyse lexicon-based  
- **Hugging Face Transformers** → Modèle RoBERTa  
- **Groq API** → Génération synthétique rapide des avis touristiques  
- **Streamlit** → Dashboard interactif  
- **Plotly** → Visualisations modernes et interactives  
- **Streamlit Community Cloud** → Déploiement gratuit

---

## 📊 Données

Les avis ont été générés **synthétiquement** à l'aide de l'**API Groq** (modèle Llama 3) pour simuler des commentaires réalistes en anglais sur Marrakech (hôtels, riads, souks, médina, accueil, nourriture, etc.).  

Le dataset contient environ 6 282 avis entre les trois classes (Positive, Negative, Neutral).

---

## 🌐 Déploiement

Le dashboard est déployé publiquement sur Streamlit Community Cloud :  

🔗 **Lien du Dashboard** : https://manalhajjaji-marrakech-reviews-appstreamlit-app-k6kg5q.streamlit.app/

*(N'hésitez pas à remplacer par votre lien réel une fois déployé !)*

---

## 🚀 Comment Exécuter Localement

### 1. Cloner le projet

```bash
git clone https://github.com/manalhajjaji/marrakech-reviews.git
cd marrakech-reviews
```

### 2. Créer un environnement virtuel

```bash
# Créer l'environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate  # sur Linux/Mac
# ou
venv\Scripts\activate  # sur Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Groq

Pour utiliser la génération de données synthétiques, vous devez obtenir une clé API Groq :

1. Créez un compte sur [Groq Console](https://console.groq.com/)
2. Générez une clé API depuis votre dashboard
3. Créez un fichier `.env` à la racine du projet :

```bash
# Créer le fichier .env
touch .env  # sur Linux/Mac
# ou créez-le manuellement sur Windows
```

4. Ajoutez votre clé API dans le fichier `.env` :

```env
GROQ_API_KEY=votre_clé_api_ici
```

### 5. Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

---



Projet réalisé avec passion pour explorer les différentes approches en analyse de sentiments et démontrer qu'une **baseline bien conçue peut parfois surpasser les modèles les plus avancés** lorsqu'elle est parfaitement adaptée au domaine.

---

**Projet réalisé dans un cadre académique par: Manal Hajjaji** 🇲🇦✨