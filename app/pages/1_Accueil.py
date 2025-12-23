import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Accueil", page_icon="🕌", layout="wide")

image_path = "../App/assets/marrakech.jpg"

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, use_container_width=True)

    st.markdown("""
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                background-color: rgba(0,0,0,0.65); padding: 25px 40px; border-radius: 20px;
                text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.6);">
        <h1 style="color: white; margin:0; font-size: 3.5rem; text-shadow: 2px 2px 8px black;">
            Analyse des Sentiments<br>des Touristes à Marrakech
        </h1>
        <h3 style="color: #4ECDC4; margin:10px 0 0 0; font-weight: 300;">
            Projet IDSCC 3ème année • ENSA Oujda • 2025-2026
        </h3>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error(f"Image non trouvée ! Chemin attendu : {image_path}")

#st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

st.markdown("""
## Objectif du projet
Développer un système complet d’**analyse de sentiments** sur les avis touristiques de **Marrakech**  
à partir d’un dataset contenant des avis synthétiques générées à l'aide de Groq.

### Points forts
- Comparaison **VADER vs RoBERTa vs un modèle de régression logistique**
- Visualisation des sentiments globaux et par modèle
- Évaluation des performances uniquement sur le **test set** pour une comparaison équitable
""")

st.success("Utilise le menu à gauche pour explorer les résultats par modèle !")
