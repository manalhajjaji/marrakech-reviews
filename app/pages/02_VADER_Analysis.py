import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

st.set_page_config(page_title="VADER", page_icon="💚", layout="wide")
st.title(" Analyse de Sentiments avec VADER")
st.caption("Évaluation des avis touristiques de Marrakech à l’aide d’un modèle lexicon-based")


# Chargement des fichiers
df_vader_full = pd.read_csv("../data/processed/results_vader.csv")
df_vader_test = pd.read_csv("../data/processed/results_vader_test.csv")

# Section explication générale sur VADER
st.markdown("## 🔎 Présentation du modèle VADER")


st.markdown("""
VADER (Valence Aware Dictionary and sEntiment Reasoner) est un outil d'analyse de sentiments simple mais puissant, spécialement conçu pour les textes courts et expressifs comme les avis en ligne.
""")

# Expander pour le fonctionnement détaillé
with st.expander("🔍 Comment ça marche ?"):
    st.markdown("""
    VADER utilise un **lexique de plus de 7 500 mots** évalués par des humains sur une échelle de -4 à +4.  
    Il applique aussi des **règles grammaticales** pour comprendre l'intensité et la polarité des phrases.
    
    **Quelques subtilités prises en compte :**
    - ❌ **Négation** : "pas bon" devient négatif
    - 📈 **Intensificateurs** : "très bon" augmente le score positif
    - ❗ **Ponctuation et majuscules** : "BON !!!" → plus positif
    - 😄 **Emojis, argot et acronymes** : ":)", "lol", "💘"
    - ⚖️ **Conjonctions contrastives** : "bon, mais mauvais" est analysé correctement
    """)

# Expander pour les scores et résultats
with st.expander("📊 Que produit VADER ?"):
    st.markdown("""
    Pour chaque texte, VADER calcule :  
    - **Score composé** entre -1 et +1
    - Proportions de **positif, négatif et neutre**
    
    Cela permet de voir rapidement si un avis est globalement positif, négatif ou neutre, même avec des phrases informelles ou pleines d'émotion.
    """)

# Expander pour forces et limites
with st.expander("💡 Forces et limites"):
    st.markdown("""
    **Forces :**
    - Excellent pour les textes courts et expressifs
    - Prend en compte emojis, ponctuation, argot  
    - Simple et rapide à utiliser
    
    **Limites :**
    - Moins précis sur les phrases longues et complexes
    - Les sentiments ambigus ou subtils peuvent être mal classés
    - Peut être complété par des modèles avancés comme **RoBERTa** pour de meilleurs résultats
    """)

st.markdown("VADER est donc idéal pour une première analyse rapide des avis touristiques sur Marrakech, tout en pouvant être combiné avec des modèles plus sophistiqués pour les cas plus subtils.")


st.markdown("## 📊 Analyse globale des sentiments")


st.markdown("### Répartition des prédictions sur l’ensemble du dataset")

# Use Plotly for bar chart with colors and smaller size
col1, col2 = st.columns(2)

with col1:
    # Bar chart
    sent_count = df_vader_full['vader_sentiment'].value_counts().reset_index()
    sent_count.columns = ['Sentiment', 'Count']
    fig_bar = px.bar(sent_count, x='Sentiment', y='Count', color='Sentiment',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                     title="Bar Chart",
                     width=500, height=400)
    st.plotly_chart(fig_bar, use_container_width=False)

with col2:
    # Ajout : Pie chart
    fig_pie = px.pie(sent_count, names='Sentiment', values='Count', color='Sentiment',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                     title="Pie Chart (Proportions)")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("## 🧪 Évaluation des performances sur le jeu de test")

st.markdown("""
Cette section évalue la performance de VADER sur l'ensemble de test. Nous utilisons des métriques comme la précision, le rappel, le F1-score et la matrice de confusion pour mesurer l'exactitude des prédictions par rapport aux labels vrais.
""")

# Calcul des métriques
true_labels = df_vader_test['sentiment_label']
pred_labels = df_vader_test['vader_sentiment']
accuracy = accuracy_score(true_labels, pred_labels)
report_dict = classification_report(true_labels, pred_labels, output_dict=True)

# Convert classification report to DataFrame for table display
report_df = pd.DataFrame(report_dict).transpose().round(2)
report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

# Style the table with colors
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

# Matrice de confusion avec Plotly heatmap pour couleurs et taille contrôlée
cm = confusion_matrix(true_labels, pred_labels, labels=['Positive', 'Negative', 'Neutral'])
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=['Positive', 'Negative', 'Neutral'],
    y=['Positive', 'Negative', 'Neutral'],
    colorscale='Blues',
    showscale=True
)
fig_cm.update_layout(title="Matrice de Confusion - VADER (Test Set)", width=500, height=400)  # Smaller size
st.plotly_chart(fig_cm, use_container_width=False)

# Interprétation dynamique des résultats
st.markdown("## 📝 Interprétation et analyse des résultats")

st.markdown(f"""
VADER atteint une accuracy globale de **{accuracy:.2%}** sur l'ensemble de test.

- **Performance par classe :**  
  - Positif : Précision de {report_dict['Positive']['precision']:.2f}, Rappel de {report_dict['Positive']['recall']:.2f}, F1-score de {report_dict['Positive']['f1-score']:.2f}. VADER identifie correctement les avis très expressifs et positifs.  
  - Négatif : Précision de {report_dict['Negative']['precision']:.2f}, Rappel de {report_dict['Negative']['recall']:.2f}, F1-score de {report_dict['Negative']['f1-score']:.2f}. Bonne sensibilité aux négations et intensificateurs.  
  - Neutre : Précision de {report_dict['Neutral']['precision']:.2f}, Rappel de {report_dict['Neutral']['recall']:.2f}, F1-score de {report_dict['Neutral']['f1-score']:.2f}. Les avis neutres ou modérés sont souvent mal classés, car VADER est optimisé pour les polarités fortes.

Dans ce projet, VADER obtient la **plus faible performance** parmi les trois approches (59.43%).  
Bien qu’il soit excellent pour les textes courts et très expressifs (emojis, ponctuation, majuscules), il peine sur les avis touristiques plus nuancés ou longs typiques de Marrakech.  
Cela montre les limites d’une approche purement lexicon-based face à des modèles entraînés sur le dataset cible.
""")