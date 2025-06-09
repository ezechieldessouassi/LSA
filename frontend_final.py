import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
from streamlit_option_menu import option_menu
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from deep_translator import GoogleTranslator

import json
import re
import nltk
import joblib
import requests
import streamlit as st

import gdown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

st.set_page_config(layout="wide", page_icon="🎬", page_title="LSA")

# Fichiers sur Google Drive
FILES = {
    "tfidf_vectorizer.joblib": "1xiTMlCp_ceoMYJ68GIBz-MErDCoQVZYI",
    "lsa_model.joblib": "1H3qa870LcA7noI7HEhx9ExBHFzgIDbmh",
    "lsa_data.npz": "1w-RrDtZbHgllJXHhO9bWis75wAkoo0rH"
}

# Téléchargement auto si fichier absent
def download_all_artifacts():
    for filename, file_id in FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            st.info(f"Téléchargement de {filename} en cours…")
            gdown.download(url, filename, quiet=False)
        else:
            print(f"{filename} déjà présent. Téléchargement ignoré.")

# Appel du téléchargement
download_all_artifacts()

# Chemins
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
LSA_MODEL_PATH  = "lsa_model.joblib"
LSA_DATA_PATH   = "lsa_data.npz"

# Traduction
translator = Translator()


# Chargement des artefacts
@st.cache_resource
def load_lsa_artifacts():
    for f in [VECTORIZER_PATH, LSA_MODEL_PATH, LSA_DATA_PATH]:
        if not os.path.isfile(f):
            st.error(f"Artefact manquant : {f}")
            st.stop()
    vec = joblib.load(VECTORIZER_PATH)
    lsa = joblib.load(LSA_MODEL_PATH)
    data = np.load(LSA_DATA_PATH, allow_pickle=True)
    return vec, lsa, data["X_lsa"], data["titles"]

vectorizer, lsa_model, X_lsa, titles_ref = load_lsa_artifacts()

def recommend_general(user_title: str, top_n: int = 10):
    tfidf_ut = vectorizer.transform([user_title])
    lsa_ut = lsa_model.transform(tfidf_ut)
    sims = cosine_similarity(lsa_ut, X_lsa).flatten()
    idxs = np.argsort(sims)[::-1][:top_n]
    return [(titles_ref[i], sims[i]) for i in idxs]


# CSS pour améliorer l'apparence
st.markdown("""
    <style>
        /* Titre principal */
        h1 {
            font-family: 'Arial', sans-serif;
            color: #1f77b4;
            font-size: 40px;
            text-align: center;
            font-weight: bold;
        }
        
        /* Sous-titres */
        h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #ff6f61;
            font-size: 25px;
            font-weight: bold;
        }
        
        /* Texte général */
        p, li, span {
            font-family: 'Nunito SemiBold', sans-serif;
            color: #333;
            font-size: 13x;
            line-height: 1.6;
        }
        
        /* Texte dans la barre latérale */
        .sidebar .sidebar-content {
            font-family: 'Poppins', sans-serif;
            color: #4d4d4d;
        }

        /* Couleur des liens */
        a {
            color: #0073e6;
            text-decoration: none;
        }

        /* Ajouter un survol aux liens */
        a:hover {
            color: #ff6f61;
        }
        
        /* Barre de progression */
        .stProgress {
            height: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# CSS
st.markdown("""
<style>
body, .main { background-color: #F8F8FF}
.header {
  background-color: #8B4513; padding: 15px 0; border-radius: 8px;
  color: white; margin-bottom: 20px;
}
.header .logo-left, .header .logo-right { width: 180px !important; margin: auto; }
.header .title { text-align: center; font-size: 2.5rem; margin: 0; line-height: 180px; }
.header:after { content: ""; display: block; clear: both; }
.team-card { display: flex; align-items: center; margin-bottom: 10px; }
.team-card .indicator {
  width: 12px; height: 12px; background-color: #4B0082;
  margin-right: 10px; border-radius: 2px;
}
.team-card .info { font-size: 1.1rem; }
.stButton>button {
  background-color: #4B0082; color: white; border-radius: 5px;
  padding: 0.5em 1.5em;
}
.stButton>button:hover { background-color: #6A0DAD; }
.section {
  background-color: #FFFFFF; padding: 15px; border-radius: 8px;
  box-shadow: 1px 1px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Barre latérale de navigation
#page = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "📽️ Modèle"])

# === PAGE ACCUEIL ========================================================================
#if page == "🏠 Accueil":
def page_accueil():
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    col_l, col_t, col_r = st.columns([1, 7, 1])
    with col_l:
        st.image("ise.jpg", width=160)
    with col_t:
        st.markdown('<h1 class="title">Recommandation de Films LSA</h1>', unsafe_allow_html=True)
    with col_r:
        st.image("eneam.jpg", width=160)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📘 Bienvenue", expanded=True):
        st.markdown("""<h3 style='font-family: "Trebuchet MS", sans-serif; font-weight: bold; color: #B8860B;'>
        Bienvenue sur l'interface de recommandation de films par similarité sémantique (modèle LSA)
        </h3>""", unsafe_allow_html=True)
        st.markdown("""
        1. 🎥 Entrez un **titre de film**
        2. 🌐 Choisissez sa **langue** (en / fr)
        3. 🔎 Cliquez sur **Recommander**
        4. 📊 Explorez les 10 films les plus similaires
        5. 💾 Téléchargez les résultats
        """)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 👥 Membres du groupe")
    members = [
        {"nom": "Peace AHOUANSE",     "rôle": "Data Scientist"},
        {"nom": "Brilland BABA",      "rôle": "Front-end"},
        {"nom": "Ezéchiel DESSOUASSI","rôle": "Ingénieur ML"},
        {"nom": "Térence KPADADONOU", "rôle": "Back-end"},
    ]
    col_a, col_b = st.columns(2)
    for idx, col in enumerate([col_a, col_b]):
        with col:
            for m in members[idx*2:(idx+1)*2]:
                st.markdown(f"""
                <div class="team-card">
                  <div class="indicator"></div>
                  <div class="info"><strong>{m["nom"]}</strong> — {m["rôle"]}</div>
                </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.title("🧠 Description du Modèle de Recommandation")

    st.header("🎯 1. Objectif du Projet")
    st.markdown("""
    L’objectif principal est de concevoir un système de recommandation de films capable de proposer des titres similaires à une requête utilisateur, en s’appuyant sur l’analyse sémantique des contenus textuels (résumés, slogans, genres).  
    Le système est **multilingue**, intègre la détection et traduction de langues, et mesure la performance via des indicateurs de **précision**.
    """)

    st.header("📊 2. Nature et Source des Données")
    st.markdown("""
    Les données sont issues d’une base téléchargée depuis **Kaggle** : `movies_metadata.csv`.  
    Cette base contient des **métadonnées sur les films**, dont seuls les champs essentiels ont été retenus :
    - `original_title`
    - `overview`
    - `genres`
    """)

    st.header("🧹 3. Chargement et Préparation des Données")
    st.markdown("""
    Le processus commence par un nettoyage :
    - Extraction structurée des genres (depuis une chaîne JSON)
    - Nettoyage des résumés : minuscules, ponctuation, mots vides, lemmatisation

    > 🔎 Ce nettoyage est **essentiel** pour garantir la qualité de l’analyse sémantique.
    """)

    st.header("🧮 4. Construction du Modèle LSA (Latent Semantic Analysis)")
    st.subheader("📥 a. Vérification des Données Sources")
    st.markdown("Une vérification préalable garantit que les données prétraitées sont bien présentes et fiables.")

    st.subheader("🧾 b. Vectorisation via TF-IDF")
    st.markdown("""
    Les résumés sont transformés en vecteurs numériques selon le schéma **TF-IDF** :
    - Renforce les mots rares mais significatifs
    - Réduit l'impact des mots très fréquents et peu informatifs
    """)

    st.subheader("🔻 c. Réduction de Dimension via LSA (SVD)")
    st.markdown("""
    Une **décomposition en valeurs singulières (SVD)** est appliquée à la matrice TF-IDF :
    - Réduit la dimension de l’espace
    - Révèle les **structures thématiques sous-jacentes**
    - Permet de **projeter** les films dans un espace sémantique réduit
    """)

    st.subheader("💾 d. Sauvegarde des Objets Modélisés")
    st.markdown("Les objets TF-IDF, SVD, corpus vectorisés et titres sont **sérialisés** pour un rechargement rapide.")

    st.subheader("📡 e. Mise en Place du Système de Recommandation")
    st.markdown("""
    Le système fonctionne en trois étapes :
    1. Traduction (si besoin)
    2. Vectorisation du texte utilisateur
    3. Calcul des **similarités cosinus** avec les films du corpus

    → Les films les plus proches sont retournés avec un **score de similarité**.
    """)

    st.subheader("✅ f. Tests de Validation")
    st.markdown("Des tests ont été réalisés avec des **titres connus** pour évaluer la cohérence des suggestions.")

    st.header("📘 5. Présentation de la Méthode LSA")
    st.markdown("""
    La **Latent Semantic Analysis (LSA)** est une méthode NLP qui :
    - Révèle les **relations sémantiques latentes**
    - Se base sur une **décomposition SVD** de la matrice TF-IDF
    """)

    st.subheader("🧾 a. Constitution de la Matrice TF-IDF")
    st.markdown("""
    - Chaque ligne = un mot du vocabulaire  
    - Chaque colonne = une description de film  
    - Chaque cellule = poids TF-IDF du mot dans le film
    """)

    st.subheader("🔻 b. Réduction avec la SVD")
    st.latex(r"A \approx U_k \Sigma_k V_k^T")
    st.markdown("""
    - A : matrice TF-IDF  
    - U_k : vecteurs propres des termes  
    - Σ_k : valeurs singulières  
    - V_k^T : vecteurs propres des documents  
    → **On garde k composantes principales** pour capter l’essentiel de l’information sémantique.
    """)

    st.subheader("🧭 c. Interprétation de l’Espace LSA")
    st.markdown("""
    - Documents proches = contenus similaires  
    - Mots proches = contextes sémantiques proches  
    - Recommandation robuste même en l’absence de termes communs
    """)

    st.success("Cette méthode permet une **recommandation sémantique intelligente**, bien plus pertinente qu’un simple appariement de mots-clés.")


# === PAGE MODÈLE =========================================================================
def modele ():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Recherche")
    user_input = st.text_input("🎥 Titre du film", placeholder="e.g. Titanic")
    lang = st.selectbox("🌐 Langue", ["en", "fr"])
    rec_button = st.button("Recommander")
    st.markdown("</div>", unsafe_allow_html=True)

    if rec_button:
        if not user_input.strip():
            st.warning("Veuillez saisir un titre de film.")
        else:
            with st.spinner("Recherche en cours…"):
                try:
                    src = detect(user_input)
                    if src != lang:
                        query = translator.translate(user_input, src=src, dest=lang).text
                    else:
                        query = user_input
                except:
                    query = user_input
                raw = recommend_general(query, top_n=10)
                df_res = pd.DataFrame(raw, columns=["Titre", "Score de similarité"])
                df_res["Score de similarité"] = df_res["Score de similarité"].map(lambda x: f"{x:.3f}")

            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.success(f"Top 10 recommandations pour « {user_input} »")
            st.table(df_res)

            col_csv, col_xlsx, _ = st.columns(3)
            with col_csv:
                st.download_button("💾 CSV", data=df_res.to_csv(index=False).encode('utf-8'),
                                   file_name="recommandations.csv", mime="text/csv")
            with col_xlsx:
                from io import BytesIO
                buffer = BytesIO()
                df_res.to_excel(buffer, index=False, sheet_name="Recommandations")
                buffer.seek(0)
                st.download_button("📊 Excel", data=buffer, file_name="recommandations.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.markdown("</div>", unsafe_allow_html=True)



# Pied de page
st.markdown("---")
st.markdown("© 2025 Projet LSA • ISE 2 ENEAM • Tous droits réservés")


with st.sidebar:
    selected_page = option_menu(
        "Menu Principal",  # Titre du menu
        ["Accueil", "Modèle", "Classification par genres","OOb"],  # Noms des pages
        icons=['house', 'play', 'film','play'],  # Icônes des pages
        menu_icon="cast",  # Icône du menu principal
        default_index=0,  # Page par défaut sélectionnée
    )


def modele_b():
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import requests
    import streamlit as st
    from langdetect import detect
    from sklearn.metrics.pairwise import cosine_similarity
    from googletrans import Translator

    # Constantes pour les chemins
    VECTORIZER_PATH = "tfidf_vectorizer.joblib"
    LSA_MODEL_PATH  = "lsa_model.joblib"
    LSA_DATA_PATH   = "lsa_data.npz"

    # 7. Recherche
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Recherche")
    user_input = st.text_input("🎥 Titre du film", placeholder="e.g. Titanic")
    lang = st.selectbox("🌐 Langue", ["en", "fr"])
    rec_button = st.button("Recommander")
    st.markdown("</div>", unsafe_allow_html=True)

    # 8. Chargement des artefacts LSA
    @st.cache_resource
    def load_lsa_artifacts():
        for f in [VECTORIZER_PATH, LSA_MODEL_PATH, LSA_DATA_PATH]:
            if not os.path.isfile(f):
                st.error(f"Artefact manquant : {f}")
                st.stop()
        vec = joblib.load(VECTORIZER_PATH)
        lsa = joblib.load(LSA_MODEL_PATH)
        data = np.load(LSA_DATA_PATH, allow_pickle=True)
        return vec, lsa, data["X_lsa"], data["titles"]

    vectorizer, lsa_model, X_lsa, titles_ref = load_lsa_artifacts()
    translator = Translator()

    # 9. Récupérer l'URL du poster via TMDb
    TMDB_API_KEY = "e63f0c5b3c1b67fc1b56421f3a0172c2"
    def get_poster_url(title: str) -> str:
        try:
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "query": title},
                timeout=5
            ).json()
            if resp.get("results"):
                path = resp["results"][0].get("poster_path")
                if path:
                    return "https://image.tmdb.org/t/p/w200" + path
        except:
            pass
        return ""

    # 10. Fonction de recommandation
    def recommend_general(user_title: str, top_n: int = 10):
        try:
            src = detect(user_title)
            if src != lang:
                ut = translator.translate(user_title, src=src, dest=lang).text
            else:
                ut = user_title
        except:
            ut = user_title

        tfidf_ut = vectorizer.transform([ut])
        lsa_ut = lsa_model.transform(tfidf_ut)
        sims = cosine_similarity(lsa_ut, X_lsa).flatten()
        idxs = np.argsort(sims)[::-1][:top_n]

        rows = []
        for i in idxs:
            title_i = titles_ref[i]
            score_i = sims[i]
            poster = get_poster_url(title_i)
            rows.append({
                "Poster": f"![]({poster})" if poster else "",
                "Titre": title_i,
                "Score de similarité": f"{score_i:.3f}",
                "PosterURL": poster
            })
        return pd.DataFrame(rows)

    # 11. Affichage & export
    if rec_button:
        if not user_input.strip():
            st.warning("Veuillez saisir un titre de film.")
        else:
            with st.spinner("Recherche en cours…"):
                df_res = recommend_general(user_input, top_n=10)

            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.success(f"Top 10 recommandations pour « {user_input} »")

            st.write(df_res.to_markdown(index=False), unsafe_allow_html=True)

            st.markdown("### 📥 Télécharger les recommandations")
            col_csv, col_xlsx, col_json = st.columns(3)

            with col_csv:
                st.download_button(
                    "💾 CSV",
                    data=df_res.drop(columns=["Poster"]).to_csv(index=False).encode("utf-8"),
                    file_name="recommandations.csv",
                    mime="text/csv",
                )
            with col_xlsx:
                from io import BytesIO
                buf = BytesIO()
                df_res.drop(columns=["Poster"]).to_excel(buf, index=False, sheet_name="Reco")
                buf.seek(0)
                st.download_button(
                    "📊 Excel",
                    data=buf,
                    file_name="recommandations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col_json:
                st.download_button(
                    "🗄 JSON",
                    data=df_res.drop(columns=["Poster"]).to_json(orient="records", force_ascii=False).encode("utf-8"),
                    file_name="recommandations.json",
                    mime="application/json",
                )

            st.markdown("</div>", unsafe_allow_html=True)

def modele2():
    # =============================================================================
    # Application Streamlit : Recommandation de films (overview + genres + images)
    # =============================================================================


    
   # nltk.download("stopwords")
    #nltk.download("wordnet")

    # === Paramètres
    TMDB_API_KEY = "e63f0c5b3c1b67fc1b56421f3a0172c2"

    # === Chargement des données
    @st.cache_data
    def load_data():
        df0 = pd.read_csv("movies_metadata.csv", low_memory=False)
        df = df0[["original_title", "overview", "genres"]].copy()

        def extract_genres(genre_str):
            try:
                js = genre_str.replace("'", '"')
                items = json.loads(js)
                return "|".join([g["name"] for g in items if "name" in g])
            except:
                return ""

        df["genres"] = df["genres"].astype(str).apply(extract_genres)
        df = df[df["genres"].str.strip() != ""].dropna(subset=["overview"]).reset_index(drop=True)

        stop_en = set(stopwords.words("english"))
        lemm = WordNetLemmatizer()

        def clean_text(text):
            text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
            tokens = [lemm.lemmatize(w) for w in text.split() if w not in stop_en and len(w) > 2]
            return " ".join(tokens)

        df["clean_overview"] = df["overview"].apply(clean_text)

        return df

    df = load_data()

    # === Modèle TF-IDF + LSA
    @st.cache_resource
    def build_model(df):
        corpus = df["clean_overview"].tolist()
        titles = df["original_title"].tolist()

        tfidf = TfidfVectorizer(max_df=0.8, min_df=5, sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(corpus)

        lsa = TruncatedSVD(n_components=100, random_state=42)
        X_lsa = lsa.fit_transform(X_tfidf)

        return tfidf, lsa, X_lsa, titles

    tfidf, lsa, X_lsa, titles = build_model(df)

    # === Fonction pour afficher les affiches
    def get_movie_poster_url(title):
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": TMDB_API_KEY, "query": title}
            response = requests.get(url, params=params)
            data = response.json()
            if data["results"]:
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return "https://image.tmdb.org/t/p/w500" + poster_path
            return None
        except:
            return None

    # === Recommandation sans filtre
    def recommend_no_filter(user_input, top_n=5, lang="fr"):
        if lang == "fr":
            translated = GoogleTranslator(source="fr", target="en").translate(user_input)
        else:
            translated = user_input

        vect_input = tfidf.transform([translated])
        lsa_input = lsa.transform(vect_input)
        sims = cosine_similarity(lsa_input, X_lsa).flatten()
        idxs = np.argsort(sims)[::-1]

        results = []
        for i in idxs[:top_n]:
            results.append((titles[i], df.iloc[i]["genres"], sims[i]))
        return results

    # === Recommandation avec filtre
    def recommend_with_filter(user_input, genre_filter, top_n=5, lang="fr"):
        if lang == "fr":
            translated = GoogleTranslator(source="fr", target="en").translate(user_input)
        else:
            translated = user_input

        vect_input = tfidf.transform([translated])
        lsa_input = lsa.transform(vect_input)
        sims = cosine_similarity(lsa_input, X_lsa).flatten()
        idxs = np.argsort(sims)[::-1]

        results = []
        count = 0
        for i in idxs:
            genres_i = str(df.iloc[i]["genres"]).lower()
            if genre_filter.lower() in genres_i:
                results.append((titles[i], df.iloc[i]["genres"], sims[i]))
                count += 1
                if count >= top_n:
                    break
        return results

    # === Interface Streamlit
    st.title("🎬 Recommandation de films (TF-IDF + LSA)")
    st.markdown("Obtenez des suggestions basées sur les **descriptions** et **genres** de films.")

    # Choix utilisateur
    lang = st.selectbox("Langue de recherche :", ["fr", "en"], index=0)
    mode = st.radio("Mode de recherche :", ["Sans filtre de genre", "Avec filtre de genre"])

    query = st.text_input("Entrez un thème ou une description de film :", "robots et espace")
    top_n = st.slider("Nombre de recommandations :", 1, 10, 5)

    if mode == "Avec filtre de genre":
        genre_choices = sorted(set(g for row in df["genres"] for g in row.split("|")))
        genre_filter = st.selectbox("Choisissez un genre :", genre_choices)
    else:
        genre_filter = None

    # Recherche
    if st.button("🔍 Rechercher"):
        st.info("Recherche en cours...")
        if mode == "Sans filtre de genre":
            results = recommend_no_filter(query, top_n, lang)
        else:
            results = recommend_with_filter(query, genre_filter, top_n, lang)

        if not results:
            st.warning("Aucun film trouvé.")
        else:
            for title, genre, score in results:
                st.subheader(f"{title} (score : {score:.3f})")
                st.markdown(f"**Genres :** {genre}")
                poster_url = get_movie_poster_url(title)
                if poster_url:
                    st.image(poster_url, width=250)
                else:
                    st.text("Image non disponible.")


if selected_page == "Accueil":
    page_accueil()
elif selected_page == "Modèle":
    modele()
elif selected_page == "Classification par genres":
    modele2()
elif selected_page=="OOb":
    modele_b()

    