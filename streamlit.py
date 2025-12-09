import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from itertools import product
from random import uniform as rd

st.set_page_config(page_title="Perceptron Visualisation", layout="wide")

# -----------------------------
# Fonctions du code tp1.py
# -----------------------------

def X(n=2, d=2):
    """Génère toutes les combinaisons de `d` indices allant de 0 à n-1, sous forme de floats."""
    return [[float(x) for x in t] for t in product(range(n), repeat=d)]

def signe(x):
    return 1 if x > 0 else -1

def f_OU(x):
    n = len(x)
    f_x = -1
    for i in range(n):
        if x[i] == 1:
            f_x = 1
            return f_x
    return f_x

def f_AND(x):
    n = len(x)
    f_x = 1
    for i in range(n):
        if x[i] == 0:
            f_x = -1
            return f_x
    return f_x

def f_XOR(x):
    result = 0
    for bit in x:
        result ^= int(bit)
    return 1 if result == 1 else -1

def L(f, X_ens):
    L_ens = []
    for n in range(len(X_ens)):
        L_ens.append([X_ens[n], f(X_ens[n][1:])])  # Le biais (x_0 = 1) est exclus
    return L_ens

def f_init_Hebb(L_ens, biais):
    p = len(L_ens)
    dim = len(L_ens[0][0])
    w = [0.0] * dim
    for k in range(p):
        w[0] = biais
        for i in range(1, dim):
            w[i] = w[i] + L_ens[k][0][i] * L_ens[k][1]
    return w

def f_init_rand(L_ens, biais, scale=1.0):
    dim = len(L_ens[0][0])
    w = [0.0] * dim
    w[0] = biais
    for i in range(1, dim):
        w[i] = rd(-scale, scale)
    return w

def count_errors(w, L_ens):
    """Compte le nombre d'erreurs d'un perceptron sur un ensemble de données"""
    errors = 0
    for x, t in L_ens:
        w_scal_x = sum(w[i] * x[i] for i in range(len(x)))
        y = 1 if w_scal_x > 0 else -1
        if y != t:
            errors += 1
    return errors

def get_min_theoretical_errors(L_ens):
    """
    Calcule le minimum théorique d'erreurs pour un ensemble de données.
    Pour XOR avec 4 points : 1 erreur minimum (non linéairement séparable)
    Pour OU/AND avec 4 points : 0 erreur (linéairement séparable)
    """
    # Si c'est XOR (4 points avec pattern spécifique)
    if len(L_ens) == 4:
        # Vérifier si c'est XOR : les points (0,0) et (1,1) ont la même classe
        # et (0,1) et (1,0) ont la même classe, mais différentes entre elles
        points = [(x[1], x[2]) for x, t in L_ens]
        labels = [t for x, t in L_ens]
        
        # Pattern XOR : (0,0)->-1, (0,1)->1, (1,0)->1, (1,1)->-1
        # Vérifier si c'est le pattern XOR
        if (0.0, 0.0) in points and (1.0, 1.0) in points:
            idx_00 = points.index((0.0, 0.0))
            idx_11 = points.index((1.0, 1.0))
            if labels[idx_00] == labels[idx_11]:  # Même classe
                if (0.0, 1.0) in points and (1.0, 0.0) in points:
                    idx_01 = points.index((0.0, 1.0))
                    idx_10 = points.index((1.0, 0.0))
                    if labels[idx_01] == labels[idx_10] and labels[idx_01] != labels[idx_00]:
                        return 1  # XOR : minimum théorique = 1 erreur
    
    # Pour les autres cas, essayer de déterminer si c'est linéairement séparable
    # Par défaut, on suppose que 0 erreur est possible (linéairement séparable)
    # mais on peut aussi retourner None pour ne pas arrêter prématurément
    return 0  # Par défaut, on vise 0 erreur (convergence complète)

def perceptron_online(w_vect, L_ens, eta, max_epochs=50):
    w_k = copy.deepcopy(w_vect)
    w_best = copy.deepcopy(w_vect)
    min_errors = count_errors(w_k, L_ens)  # Compter les erreurs initiales
    min_theoretical = get_min_theoretical_errors(L_ens)  # Minimum théorique
    stop = 0
    converged = False
    optimal_reached = False
    
    while stop < max_epochs:
        nb_mal_classe = 0
        for k in range(len(L_ens)):
            x_k = L_ens[k][0]
            t = L_ens[k][1]
            w_k_scal_x_k = sum(w_k[i] * x_k[i] for i in range(len(x_k)))
            y = 1 if w_k_scal_x_k > 0 else -1
            if y != t:
                delta_w = eta * (t - y)
                for i in range(len(x_k)):
                    w_k[i] += delta_w * x_k[i]
                nb_mal_classe += 1
        
        # Vérifier les erreurs après la mise à jour de l'époque
        current_errors = count_errors(w_k, L_ens)
        
        # Garder le meilleur perceptron (moins d'erreurs)
        if current_errors < min_errors:
            min_errors = current_errors
            w_best = copy.deepcopy(w_k)
        
        # Arrêter si on a atteint le minimum théorique
        if min_errors <= min_theoretical:
            optimal_reached = True
            converged = (min_errors == 0)
            stop += 1
            break
        
        if nb_mal_classe == 0:
            converged = True
            w_best = copy.deepcopy(w_k)
            min_errors = 0
            stop += 1
            break
        else:
            stop += 1
    
    return w_best, stop - 1, converged or optimal_reached, min_errors

def perceptron_batch(w_vect, L_ens, eta, max_epochs=50):
    w_k = copy.deepcopy(w_vect)
    w_best = copy.deepcopy(w_vect)
    min_errors = count_errors(w_k, L_ens)  # Compter les erreurs initiales
    min_theoretical = get_min_theoretical_errors(L_ens)  # Minimum théorique
    stop = 0
    converged = False
    optimal_reached = False
    
    while stop < max_epochs:
        nb_mal_classe = 0
        delta_w = np.zeros(len(w_vect))
        for k in range(len(L_ens)):
            x_k = L_ens[k][0]
            t = L_ens[k][1]
            w_k_scal_x_k = sum(w_k[i] * x_k[i] for i in range(len(x_k)))
            y = 1 if w_k_scal_x_k > 0 else -1
            if y != t:
                for i in range(len(x_k)):
                    delta_w[i] = delta_w[i] + eta * (t - y) * x_k[i]
                nb_mal_classe += 1
        w_k = list(np.array(w_k) + delta_w)
        
        # Vérifier les erreurs après la mise à jour de l'époque
        current_errors = count_errors(w_k, L_ens)
        
        # Garder le meilleur perceptron (moins d'erreurs)
        if current_errors < min_errors:
            min_errors = current_errors
            w_best = copy.deepcopy(w_k)
        
        # Arrêter si on a atteint le minimum théorique
        if min_errors <= min_theoretical:
            optimal_reached = True
            converged = (min_errors == 0)
            stop += 1
            break
        
        if nb_mal_classe == 0:
            converged = True
            w_best = copy.deepcopy(w_k)
            min_errors = 0
            stop += 1
            break
        else:
            stop += 1
    
    return w_best, stop - 1, converged or optimal_reached, min_errors

def init_perceptron_prof(N, biais_range=(-10e6, 10e6)):
    w_prof = np.zeros(N+1)
    biais_min, biais_max = biais_range
    w_prof[0] = np.random.uniform(biais_min, biais_max)
    for i in range(1, len(w_prof)):
        w_prof[i] = rd(biais_min, biais_max)
    return w_prof

def L_prof(X, w_prof):
    L_ens = []
    for x in X:
        L_ens.append([x, signe(np.dot(x, w_prof))])
    return L_ens

def init_many_perceptrons(qte, N, biais_range=(-10e6, 10e6)):
    borne_min, borne_max = biais_range
    W_many_perceptrons = np.random.uniform(borne_min, borne_max, size=(qte, N))
    W_avec_biais = np.zeros((qte, N + 1))
    biais_min, biais_max = biais_range
    for j in range(qte):
        W_avec_biais[j, 0] = np.random.uniform(biais_min, biais_max)
        W_avec_biais[j, 1:] = W_many_perceptrons[j]
    return W_avec_biais

def create_points(nbPoints, N, bornes=(-10e6, 10e6)):
    points = []
    borne_min, borne_max = bornes
    for p in range(nbPoints):
        point = [1.0]  # Le biais
        for i in range(N):
            x_i = rd(borne_min, borne_max)
            point.append(x_i)
        points.append(point)
    return points

def recouvrement(vect1, vect2):
    produit_scalaire = np.dot(vect1, vect2)
    norme_vect_1 = np.linalg.norm(vect1)
    norme_vect_2 = np.linalg.norm(vect2)
    if norme_vect_1 == 0 or norme_vect_2 == 0:
        return 0.0
    R = produit_scalaire / (norme_vect_1 * norme_vect_2)
    return R

def plot_decision(L_ens, w_k, ax=None, label=None, linewidth=2, linestyle='-', color=None):
    """Trace la frontière de décision d'un perceptron"""
    x1_vals = [x[1] for x, t in L_ens]
    x2_vals = [x[2] for x, t in L_ens]
    labels = [t for x, t in L_ens]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors_points = ['red' if t == -1 else 'blue' for t in labels]
    ax.scatter(x1_vals, x2_vals, c=colors_points, s=100, alpha=0.7)

    w0, w1, w2 = w_k[0], w_k[1], w_k[2]
    x1_min, x1_max = min(x1_vals) - 1, max(x1_vals) + 1

    if abs(w2) > 1e-10:
        x1_line = np.array([x1_min, x1_max])
        x2_line = -(w1 * x1_line + w0) / w2
    else:
        x_fixed = -w0 / w1 if abs(w1) > 1e-10 else 0.0
        x1_line = np.array([x_fixed, x_fixed])
        x2_min, x2_max = min(x2_vals), max(x2_vals)
        x2_line = np.array([x2_min - 1, x2_max + 1])

    line_color = color if color else 'black'
    ax.plot(x1_line, x2_line, linestyle=linestyle, linewidth=linewidth, 
            label=label, color=line_color)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Frontière de décision')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    if label:
        ax.legend()
    return ax

def train_perceptron(perceptron, L_ens, eta, algorithme, max_epochs=50):
    # Si c'est XOR, utiliser 200 époques au lieu de max_epochs
    min_theoretical = get_min_theoretical_errors(L_ens)
    if min_theoretical == 1:  # C'est XOR (minimum théorique = 1 erreur)
        max_epochs = 200  # Aller jusqu'à 200 époques pour XOR
    
    if algorithme == "online":
        perceptron, nbIter, converged, min_errors = perceptron_online(perceptron, L_ens, eta, max_epochs)
    elif algorithme == "batch":
        perceptron, nbIter, converged, min_errors = perceptron_batch(perceptron, L_ens, eta, max_epochs)
    return perceptron, nbIter, converged, min_errors

# -----------------------------
# Interface Streamlit
# -----------------------------

st.title('Visualisation Perceptron (Devélopper par Christ et patrice)')

# Sidebar pour les paramètres
with st.sidebar:
    st.header('⚙️ Paramètres')
    
    # Section 1: Fonction booléenne
    st.subheader('1. Fonction booléenne')
    fonction_bool = st.selectbox(
        'Fonction booléenne',
        ['OU', 'AND', 'XOR'],
        index=0
    )
    
    # Section 2: Perceptron Professeur
    st.subheader('2. Perceptron Professeur')
    use_prof = st.checkbox('Utiliser perceptron professeur', value=False)
    prof_biais_min = st.number_input('Biais min (prof)', value=-20.0, step=1.0)
    prof_biais_max = st.number_input('Biais max (prof)', value=20.0, step=1.0)
    nb_points_prof = st.number_input('Nombre de points P', value=100, min_value=10, max_value=1000, step=10)
    points_bornes_min = st.number_input('Bornes min (points)', value=-10.0, step=1.0)
    points_bornes_max = st.number_input('Bornes max (points)', value=10.0, step=1.0)
    
    # Section 3: Initialisation
    st.subheader('3. Initialisation')
    init_type = st.selectbox('Type d\'initialisation', ['Aléatoire', 'Hebb'], index=0)
    biais = st.number_input('Biais initial (w0)', value=1.0, step=0.5)
    init_scale = st.number_input('Amplitude init aléatoire', value=1.0, step=0.1)
    
    # Section 4: Entraînement
    st.subheader('4. Entraînement')
    algorithme = st.radio('Algorithme', ['online', 'batch'], index=0)
    eta = st.slider('Taux d\'apprentissage (eta)', 0.01, 1.0, 0.14, 0.01)
    max_epochs = st.slider('Max epochs', 1, 200, 50, 1)
    try_multiple_init = st.checkbox('Essayer plusieurs initialisations (recommandé pour XOR)', value=False)
    nb_trials = st.number_input('Nombre d\'essais', value=10, min_value=1, max_value=50, step=1) if try_multiple_init else 1
    
    # Section 5: Visualisation multiple
    st.subheader('5. Visualisation multiple')
    show_multiple = st.checkbox('Afficher plusieurs perceptrons', value=False)
    nb_perceptrons = st.number_input('Nombre de perceptrons', value=5, min_value=1, max_value=20, step=1)

# -----------------------------
# Génération des données
# -----------------------------

N = 2
X_ens = X(2, N)
X_ens = [[1.0] + x for x in X_ens]

# Créer L_ens selon la fonction choisie
if fonction_bool == 'OU':
    L_ens = L(f_OU, X_ens)
elif fonction_bool == 'AND':
    L_ens = L(f_AND, X_ens)
elif fonction_bool == 'XOR':
    L_ens = L(f_XOR, X_ens)

# Si on utilise le perceptron professeur
if use_prof:
    w_prof = init_perceptron_prof(N, (prof_biais_min, prof_biais_max))
    points = create_points(nb_points_prof, N, (points_bornes_min, points_bornes_max))
    L_ens = L_prof(points, w_prof)
    st.sidebar.success(f'Perceptron professeur créé: {np.round(w_prof, 3)}')

# -----------------------------
# Affichage des données
# -----------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(' Jeu de données')
    df_data = pd.DataFrame({
        'x (avec biais)': [str(x) for x, t in L_ens],
        'label': [t for x, t in L_ens]
    })
    st.dataframe(df_data, width='stretch')
    
    st.subheader('Statistiques')
    st.write(f"Nombre de points: {len(L_ens)}")
    st.write(f"Classe +1: {sum(1 for _, t in L_ens if t == 1)}")
    st.write(f"Classe -1: {sum(1 for _, t in L_ens if t == -1)}")

with col2:
    st.subheader(' Visualisation initiale')
    # Initialiser le perceptron
    if init_type == 'Aléatoire':
        w0 = f_init_rand(L_ens, biais, scale=init_scale)
    else:
        w0 = f_init_Hebb(L_ens, biais)
    
    fig_init, ax_init = plt.subplots(figsize=(6, 6))
    plot_decision(L_ens, w0, ax=ax_init, label='Initial')
    st.pyplot(fig_init)
    st.write(f"**Poids initiaux:** {np.round(w0, 3)}")

# -----------------------------
# Entraînement
# -----------------------------

if st.button('🚀 Entraîner le perceptron', type='primary'):
    # Vérifier si c'est XOR pour afficher un message informatif
    min_theoretical = get_min_theoretical_errors(L_ens)
    if min_theoretical == 1:  # C'est XOR
        st.info("🔍 **XOR détecté** : L'apprentissage utilisera jusqu'à 200 époques et s'arrêtera dès qu'on atteint 1 erreur (minimum théorique).")
    
    with st.spinner('Entraînement en cours...'):
        best_w_final = None
        best_nb_iter = 0
        best_converged = False
        best_min_errors = len(L_ens)
        best_w0 = None
        
        # Essayer plusieurs initialisations si demandé
        trials = nb_trials if try_multiple_init else 1
        progress_bar = st.progress(0)
        
        for trial in range(trials):
            # Initialiser le perceptron
            if init_type == 'Aléatoire':
                w0 = f_init_rand(L_ens, biais, scale=init_scale)
            else:
                w0 = f_init_Hebb(L_ens, biais)
            
            # Entraîner
            w_final, nb_iter, converged, min_errors = train_perceptron(w0, L_ens, eta, algorithme, max_epochs)
            
            # Garder le meilleur résultat
            if min_errors < best_min_errors or (min_errors == best_min_errors and converged and not best_converged):
                best_w_final = w_final
                best_nb_iter = nb_iter
                best_converged = converged
                best_min_errors = min_errors
                best_w0 = w0
            
            # Si convergence atteinte, on peut s'arrêter
            if converged:
                break
            
            progress_bar.progress((trial + 1) / trials)
        
        progress_bar.empty()
        
        w_final = best_w_final
        nb_iter = best_nb_iter
        converged = best_converged
        min_errors = best_min_errors
        w0 = best_w0
        
        if try_multiple_init and trials > 1:
            st.info(f"🔍 {trials} essais effectués - Meilleur résultat conservé")
        
        # Calculer le recouvrement si on a un professeur
        R = None
        if use_prof:
            R = recouvrement(w_prof, w_final)
        
        # Calculer le minimum théorique pour l'affichage
        min_theoretical = get_min_theoretical_errors(L_ens)
        optimal_reached = (min_errors <= min_theoretical)
        
        # Afficher les résultats
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            if converged and min_errors == 0:
                st.success(f'✅ Convergence atteinte! (0 erreur)')
            elif optimal_reached and min_errors > 0:
                st.success(f'✅ Optimum théorique atteint! ({min_errors} erreur{"s" if min_errors > 1 else ""} - impossible de faire mieux)')
                st.info(f"**Erreurs:** {min_errors}/{len(L_ens)} points mal classés (minimum théorique)")
            else:
                st.warning(f'⚠️ Convergence non atteinte - Meilleur perceptron affiché')
                st.info(f"**Erreurs minimales:** {min_errors}/{len(L_ens)} points mal classés")
            st.write(f"**Nombre d'itérations:** {nb_iter}")
            st.write(f"**Poids finaux:** {np.round(w_final, 3)}")
            if R is not None:
                st.write(f"**Recouvrement R:** {R:.4f}")
        
        with col_res2:
            fig_final, ax_final = plt.subplots(figsize=(6, 6))
            plot_decision(L_ens, w_final, ax=ax_final, label='Après entraînement', linewidth=3)
            if use_prof:
                plot_decision(L_ens, w_prof, ax=ax_final, label='Professeur', 
                            linewidth=2, linestyle='--', color='red')
            st.pyplot(fig_final)
        
        # Comparaison avant/après
        st.subheader('📊 Comparaison avant/après')
        col_comp1, col_comp2 = st.columns([1, 1])
        
        with col_comp1:
            st.write("**Avant entraînement**")
            fig_before, ax_before = plt.subplots(figsize=(6, 6))
            plot_decision(L_ens, w0, ax=ax_before, label='Initial')
            st.pyplot(fig_before)
        
        with col_comp2:
            st.write("**Après entraînement**")
            fig_after, ax_after = plt.subplots(figsize=(6, 6))
            plot_decision(L_ens, w_final, ax=ax_after, label='Final')
            if use_prof:
                plot_decision(L_ens, w_prof, ax=ax_after, label='Professeur', 
                            linewidth=2, linestyle='--', color='red')
            st.pyplot(fig_after)

# -----------------------------
# Visualisation multiple
# -----------------------------

if show_multiple:
    st.markdown('---')
    st.subheader('Comparaison de plusieurs perceptrons')
    
    if st.button('Générer et entraîner plusieurs perceptrons'):
        with st.spinner('Génération et entraînement en cours...'):
            # Générer plusieurs perceptrons
            biais_range = (prof_biais_min, prof_biais_max)
            W_perceptrons = init_many_perceptrons(nb_perceptrons, N, biais_range)
            
            # Entraîner chaque perceptron
            results = []
            for i, w_init in enumerate(W_perceptrons):
                w_final, nb_iter, converged, min_errors = train_perceptron(w_init, L_ens, eta, algorithme, max_epochs)
                R = recouvrement(w_prof, w_final) if use_prof else None
                convergence_status = "✅ Oui" if converged else f"❌ Non ({min_errors} erreurs)"
                results.append({
                    'Perceptron': i+1,
                    'Poids initiaux': str(np.round(w_init, 3)),
                    'Poids finaux': str(np.round(w_final, 3)),
                    'Nb itérations': nb_iter,
                    'Convergence': convergence_status,
                    'Erreurs min': min_errors,
                    'Recouvrement R': f'{R:.4f}' if R is not None else 'N/A'
                })
            
            # Afficher le tableau de résultats
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, width='stretch')
            
            # Visualisation graphique
            fig_multi, ax_multi = plt.subplots(figsize=(10, 8))
            
            # Tracer les points
            x1_vals = [x[1] for x, t in L_ens]
            x2_vals = [x[2] for x, t in L_ens]
            labels = [t for x, t in L_ens]
            colors_points = ['red' if t == -1 else 'blue' for t in labels]
            ax_multi.scatter(x1_vals, x2_vals, c=colors_points, s=100, alpha=0.7)
            
            # Tracer chaque perceptron
            colors = plt.cm.rainbow(np.linspace(0, 1, nb_perceptrons))
            for i, w in enumerate(W_perceptrons):
                w_final, _, _, _ = train_perceptron(w, L_ens, eta, algorithme, max_epochs)
                w0, w1, w2 = w_final[0], w_final[1], w_final[2]
                x1_min, x1_max = min(x1_vals) - 1, max(x1_vals) + 1
                
                if abs(w2) > 1e-10:
                    x1_line = np.array([x1_min, x1_max])
                    x2_line = -(w1 * x1_line + w0) / w2
                else:
                    x_fixed = -w0 / w1 if abs(w1) > 1e-10 else 0.0
                    x1_line = np.array([x_fixed, x_fixed])
                    x2_min, x2_max = min(x2_vals), max(x2_vals)
                    x2_line = np.array([x2_min - 1, x2_max + 1])
                
                ax_multi.plot(x1_line, x2_line, color=colors[i], linewidth=1.5, 
                            linestyle='--', label=f'Perceptron {i+1}', alpha=0.7)
            
            # Tracer le professeur si disponible
            if use_prof:
                w0, w1, w2 = w_prof[0], w_prof[1], w_prof[2]
                x1_min, x1_max = min(x1_vals) - 1, max(x1_vals) + 1
                if abs(w2) > 1e-10:
                    x1_line = np.array([x1_min, x1_max])
                    x2_line = -(w1 * x1_line + w0) / w2
                else:
                    x_fixed = -w0 / w1 if abs(w1) > 1e-10 else 0.0
                    x1_line = np.array([x_fixed, x_fixed])
                    x2_min, x2_max = min(x2_vals), max(x2_vals)
                    x2_line = np.array([x2_min - 1, x2_max + 1])
                ax_multi.plot(x1_line, x2_line, 'r-', linewidth=3, label='Professeur')
            
            ax_multi.set_xlabel('x1')
            ax_multi.set_ylabel('x2')
            ax_multi.set_title(f'Comparaison de {nb_perceptrons} perceptrons')
            ax_multi.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_multi.grid(True, alpha=0.3)
            ax_multi.axis('equal')
            plt.tight_layout()
            st.pyplot(fig_multi)

# -----------------------------
# Footer
# -----------------------------
st.markdown('---')
st.caption('Interface de visualisation pour le TP1 - Perceptron')

