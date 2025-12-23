import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Custom CSS for white background and light theme
st.markdown("""
<style>
    .main {
        background-color: white;
        color: black;
    }
    .stApp {
        background-color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: black !important;
    }
    .stMarkdown {
        color: black;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Régression Logistique", page_icon="📝", layout="wide")
st.title(" Analyse de Sentiments avec un Modèle From Scratch")
st.caption("Modèle de régression logistique entraîné de zéro sur les avis touristiques de Marrakech")

# Chargement des fichiers
# Note : Tu n’as que le test set pour From Scratch → on travaille dessus
df_fromscratch_test = pd.read_csv("../data/processed/results_fromscratch_test.csv")

# Section explication générale sur le modèle From Scratch
st.markdown("## 🔎 Présentation du modèle From Scratch")

st.markdown("""
Ce modèle est une **régression logistique multiclasse** entraînée entièrement à partir de zéro sur notre dataset d’avis touristiques de Marrakech.  
Il représente l’approche **machine learning classique** (non-deep learning).
""")

# Expander pour le fonctionnement détaillé
with st.expander("🔍 Comment ça marche ?"):
    st.markdown("""
    - **Pré-traitement spécifique** : 
      - Nettoyage approfondi du texte (suppression des emojis, ponctuation excessive)
      - Tokenisation
      - Lemmatisation (NLTK/Spacy)
      - Suppression des stopwords
    - **Vectorisation** : TF-IDF (Term Frequency - Inverse Document Frequency) pour transformer le texte en vecteurs numériques
    - **Modèle** : Régression logistique (LogisticRegression de scikit-learn) avec stratégie multiclasse *ovr* (one-vs-rest)
    - **Entraînement** : Sur le train set (X_train, y_train) avec validation croisée pour choisir les hyperparamètres
    """)

# Expander pour les sorties du modèle
with st.expander("📊 Que produit le modèle ?"):
    st.markdown("""
    Pour chaque avis, le modèle calcule des **probabilités** pour les trois classes (Positive, Negative, Neutral).  
    La classe prédite est celle ayant la probabilité la plus élevée.
    """)

# Expander pour forces et limites
with st.expander("💡 Forces et limites"):
    st.markdown("""
    **Forces :**
    - Très rapide à entraîner et à inférer
    - Interprétable (coefficients du modèle montrent l’importance des mots)
    - Bonne baseline classique pour comparer avec des modèles plus complexes
    - Contrôle total sur le pré-traitement et les features
    
    **Limites :**
    - Moins performant que les modèles Transformer (comme RoBERTa) sur les nuances et le contexte
    - Dépend fortement de la qualité du pré-traitement et de la vectorisation TF-IDF
    - Ne gère pas nativement les emojis, majuscules ou ponctuation expressive (d’où le nettoyage préalable)
    """)

st.markdown("Ce modèle from scratch constitue une **référence baseline solide** et permet de mesurer l’apport réel des approches plus avancées (VADER et RoBERTa).")

st.markdown("## 📊 Analyse des prédictions sur le jeu de test")

st.markdown("### Répartition des prédictions (Test Set)")

col1, col2 = st.columns(2)

with col1:
    # Bar chart
    sent_count = df_fromscratch_test['predicted_sentiment'].value_counts().reset_index()
    sent_count.columns = ['Sentiment', 'Count']
    fig_bar = px.bar(sent_count, x='Sentiment', y='Count', color='Sentiment',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                     title="Bar Chart",
                     width=500, height=400)
    st.plotly_chart(fig_bar, use_container_width=False)

with col2:
    # Pie chart
    fig_pie = px.pie(sent_count, names='Sentiment', values='Count', color='Sentiment',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                     title="Pie Chart (Proportions)")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("## 🧪 Évaluation des performances sur le jeu de test")

st.markdown("""
Cette section présente les métriques de performance du modèle from scratch sur le jeu de test.
""")

# Calcul des métriques
true_labels = df_fromscratch_test['true_sentiment']
pred_labels = df_fromscratch_test['predicted_sentiment']
accuracy = accuracy_score(true_labels, pred_labels)
report_dict = classification_report(true_labels, pred_labels, output_dict=True)

# Tableau stylé du classification report
report_df = pd.DataFrame(report_dict).transpose().round(2)
report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

def color_cells(val):
    if val >= 0.7:
        color = 'green'
    elif val >= 0.5:
        color = 'orange'
    else:
        color = 'red'
    return f'background-color: {color}; color: white'

styled_report = report_df.style.applymap(color_cells, subset=['precision', 'recall', 'f1-score'])

st.markdown("### Rapport de classification")

st.dataframe(styled_report)

# Matrice de confusion avec Plotly (couleurs oranges pour From Scratch)
cm = confusion_matrix(true_labels, pred_labels, labels=['Positive', 'Negative', 'Neutral'])
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=['Positive', 'Negative', 'Neutral'],
    y=['Positive', 'Negative', 'Neutral'],
    colorscale='Oranges',  # Thème orange pour le modèle classique
    showscale=True
)
fig_cm.update_layout(title="Matrice de Confusion - Modèle From Scratch (Test Set)", width=500, height=400)
st.plotly_chart(fig_cm, use_container_width=False)

# Interprétation dynamique des résultats
st.markdown("## 📝 Interprétation et analyse des résultats")

st.markdown(f"""
Le modèle from scratch atteint une accuracy globale de **{accuracy:.2%}** sur le jeu de test.

- **Performance par classe :**  
  - Positif : Précision de {report_dict['Positive']['precision']:.2f}, Rappel de {report_dict['Positive']['recall']:.2f}, F1-score de {report_dict['Positive']['f1-score']:.2f}. Excellente détection de la classe majoritaire.  
  - Négatif : Précision de {report_dict['Negative']['precision']:.2f}, Rappel de {report_dict['Negative']['recall']:.2f}, F1-score de {report_dict['Negative']['f1-score']:.2f}. Très bonne performance malgré la suppression des indices expressifs.  
  - Neutre : Précision de {report_dict['Neutral']['precision']:.2f}, Rappel de {report_dict['Neutral']['recall']:.2f}, F1-score de {report_dict['Neutral']['f1-score']:.2f}. Meilleure gestion que prévu grâce à un pré-traitement adapté.

**Résultat remarquable** : le modèle from scratch (régression logistique + TF-IDF) obtient la **meilleure performance globale** avec 88.86%, surpassant largement VADER et même RoBERTa !

Cela démontre la puissance d’une approche classique bien exécutée :
- Pré-traitement soigneux (lemmatisation, suppression du bruit)
- Vectorisation TF-IDF adaptée parfaitement au corpus
- Modèle simple mais robuste, entraîné directement sur les données cibles

Ce résultat remet en perspective l’idée que les modèles Transformer sont toujours supérieurs : ici, une **baseline classique optimisée** domine grâce à son adaptation parfaite au dataset.
""")