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

st.set_page_config(page_title="RoBERTa", page_icon="🤖", layout="wide")
st.title(" Analyse de Sentiments avec RoBERTa")
st.caption("Évaluation des avis touristiques de Marrakech à l’aide d’un modèle Transformer pré-entraîné")

# Chargement des fichiers
df_roberta_full = pd.read_csv("../data/processed/results_roberta.csv")
df_roberta_test = pd.read_csv("../data/processed/results_roberta_test.csv")

# Section explication générale sur RoBERTa
st.markdown("## 🔎 Présentation du modèle RoBERTa")

st.markdown("""
RoBERTa (Robustly optimized BERT Pretraining Approach) est une version améliorée du célèbre modèle BERT de Google.  
Il s’agit d’un **modèle Transformer** pré-entraîné sur une très grande quantité de texte (milliards de phrases), puis fine-tuné sur des tâches d’analyse de sentiments.
""")

# Expander pour le fonctionnement détaillé
with st.expander("🔍 Comment ça marche ?"):
    st.markdown("""
    - RoBERTa repose sur une **architecture Transformer** avec attention multi-têtes qui capture les relations complexes entre les mots dans une phrase.
    - Il a été entraîné avec plus de données et des optimisations (suppression du NSP, entraînement plus long, batches plus grands) → meilleures performances que BERT.
    - Pour l’analyse de sentiments, on utilise généralement **"cardiffnlp/twitter-roberta-base-sentiment-latest"** ou un modèle similaire fine-tuné sur des avis et tweets.
    - Le modèle comprend le **contexte bidirectionnel** (il lit la phrase dans les deux sens) → très performant sur les phrases ambiguës, le sarcasme et les sentiments subtils.
    """)

# Expander pour les sorties du modèle
with st.expander("📊 Que produit RoBERTa ?"):
    st.markdown("""
    Pour chaque avis, RoBERTa retourne des **probabilités (logits)** pour chaque classe :
    - Positive
    - Negative  
    - Neutral
    
    La classe prédite est celle avec la probabilité la plus élevée.
    """)

# Expander pour forces et limites
with st.expander("💡 Forces et limites"):
    st.markdown("""
    **Forces :**
    - Très haute précision, surtout sur les sentiments nuancés et ambigus
    - Gère bien le langage informel, le sarcasme et les négations complexes
    - Performances state-of-the-art sur la plupart des benchmarks de sentiment analysis
    
    **Limites :**
    - Plus lent et gourmand en ressources que VADER (nécessite GPU pour l’inférence rapide)
    - Moins interprétable (boîte noire)
    - Sensible à la qualité du fine-tuning (le modèle utilisé doit être adapté au domaine)
    """)

st.markdown("Dans ce projet, RoBERTa représente l’approche **deep learning moderne** et sert de référence performante face aux méthodes rule-based comme VADER et au modèle from scratch.")

st.markdown("## 📊 Analyse globale des sentiments")

st.markdown("### Répartition des prédictions sur l’ensemble du dataset")

col1, col2 = st.columns(2)

with col1:
    # Bar chart
    sent_count = df_roberta_full['roberta_sentiment'].value_counts().reset_index()
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
Cette section évalue la performance de RoBERTa sur l'ensemble de test avec les métriques classiques.
""")

# Calcul des métriques
true_labels = df_roberta_test['sentiment_label']
pred_labels = df_roberta_test['roberta_sentiment']
accuracy = accuracy_score(true_labels, pred_labels)
report_dict = classification_report(true_labels, pred_labels, output_dict=True)

# Tableau stylé du classification report
report_df = pd.DataFrame(report_dict).transpose().round(2)
report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

def color_cells(val):
    if val >= 0.8:
        color = 'green'
    elif val >= 0.6:
        color = 'orange'
    else:
        color = 'red'
    return f'background-color: {color}; color: white'

styled_report = report_df.style.applymap(color_cells, subset=['precision', 'recall', 'f1-score'])

st.markdown("### Rapport de classification")

st.dataframe(styled_report)

# Matrice de confusion avec Plotly (couleurs vertes pour RoBERTa)
cm = confusion_matrix(true_labels, pred_labels, labels=['Positive', 'Negative', 'Neutral'])
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=['Positive', 'Negative', 'Neutral'],
    y=['Positive', 'Negative', 'Neutral'],
    colorscale='Greens',  # Thème vert pour RoBERTa
    showscale=True
)
fig_cm.update_layout(title="Matrice de Confusion - RoBERTa (Test Set)", width=500, height=400)
st.plotly_chart(fig_cm, use_container_width=False)

# Interprétation dynamique des résultats
st.markdown("## 📝 Interprétation et analyse des résultats")

st.markdown(f"""
RoBERTa atteint une accuracy globale de **{accuracy:.2%}** sur l'ensemble de test.

- **Performance par classe :**  
  - Positif : Précision de {report_dict['Positive']['precision']:.2f}, Rappel de {report_dict['Positive']['recall']:.2f}, F1-score de {report_dict['Positive']['f1-score']:.2f}. Très bonne détection des avis positifs.  
  - Négatif : Précision de {report_dict['Negative']['precision']:.2f}, Rappel de {report_dict['Negative']['recall']:.2f}, F1-score de {report_dict['Negative']['f1-score']:.2f}. Bonne gestion du sarcasme et des négations complexes.  
  - Neutre : Précision de {report_dict['Neutral']['precision']:.2f}, Rappel de {report_dict['Neutral']['recall']:.2f}, F1-score de {report_dict['Neutral']['f1-score']:.2f}. Meilleure compréhension des nuances que VADER.

Surprenant dans ce projet : RoBERTa (77.72%) est **dépassé par le modèle from scratch**.  
Cela peut s’expliquer par plusieurs facteurs :
- Le modèle RoBERTa utilisé n’était peut-être pas parfaitement adapté au domaine touristique ou au style des avis (fine-tuning général).
- Le dataset contient des avis en anglais avec un vocabulaire spécifique à Marrakech que le modèle from scratch, entraîné directement dessus, capture mieux via TF-IDF.
- RoBERTa reste supérieur sur les cas ambigus et contextuels, mais le pré-traitement + régression logistique s’avère ici plus efficace globalement.
""")