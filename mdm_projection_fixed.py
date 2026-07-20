# -*- coding: utf-8 -*-
"""
mdm_projection.py  (versao corrigida)
Duas funcoes que faltam para fechar a proposta do artigo
"Evaluating brain group structure methods using hierarchical dynamic models"
(Costa et al., Pattern Recognition 155, 2024, 110687), a serem integradas ao
pacote MP (https://github.com/ProfNascimento/MP).

1) sparse_mdm_dissimilarity  -> constroi a matriz de separacao d(i,j) do MDM
   (log Bayes factor) com suporte a matriz ESPARSA (so os pares avaliados).
2) plot_projections          -> plota varias projecoes multidimensionais
   (MDS, t-SNE, UMAP, ForceScheme, LAMP, PCoA) + o dendrograma do artigo,
   a partir da matriz de distancias gerada em (1), tratando esparsidade.

Requisitos: numpy, scipy, scikit-learn, matplotlib
Opcionais : umap-learn (UMAP), mppy (ForceScheme/LAMP)

Correcoes aplicadas nesta versao:
  - _pcoa agora e o MDS classico correto (V*sqrt(lambda)), nao PCA(B).
  - t-SNE/UMAP avisam ao cair da esparsa para a densa (fallback).
  - sparse_mdm_dissimilarity avisa quando d(i,j) e fortemente negativo.
  - complete_sparse_distance documenta a semantica de zero explicito.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


# ----------------------------------------------------------------------------
# FUNCAO 1 - matriz de dissimilaridade esparsa derivada do MDM
# ----------------------------------------------------------------------------
def sparse_mdm_dissimilarity(n_subjects, lpl_self, pairs, *,
                             clip_negative=True, as_sparse=True, dtype=float):
    r"""
    Constroi a matriz de separacao d(i,j) do MDM a partir das LPLs.

    Formula do artigo (Secao 3):
        d(i,j) = LPL_i(M_i) + LPL_j(M_j) - LPL_i(m_ij) - LPL_j(m_ij)

    onde M_i e a rede individual do sujeito i e m_ij e a rede conjunta
    (pooled) do par (i, j).

    Parametros
    ----------
    n_subjects : int
        Numero de sujeitos S (dimensao S x S da matriz).
    lpl_self : array (S,)
        LPL_i(M_i) -- score da rede individual de cada sujeito.
    pairs : iterable de (i, j, lpl_i_mij, lpl_j_mij)
        SOMENTE os pares realmente avaliados via MDM-IPA. Passar so um
        subconjunto de todos os S*(S-1)/2 pares e o que gera a esparsidade.
    clip_negative : bool
        Trunca pequenos negativos (ruido numerico) em 0. d(i,j) e >= 0.
    as_sparse : bool
        Se True retorna scipy.sparse.csr_matrix; se False retorna np.ndarray.
    dtype : tipo numerico.

    Retorna
    -------
    D : csr_matrix ou ndarray (S x S), simetrica, diagonal 0.
    mask : csr_matrix booleana (S x S)
        True nas posicoes REALMENTE computadas. Necessaria porque, numa
        matriz esparsa, "ausente" (par nao avaliado) e diferente de
        "distancia zero" (sujeitos identicos).
    """
    lpl_self = np.asarray(lpl_self, dtype=dtype).ravel()
    if lpl_self.shape[0] != n_subjects:
        raise ValueError("lpl_self deve ter tamanho n_subjects.")
    rows, cols, vals = [], [], []
    seen = set()
    for tup in pairs:
        i, j, lpl_i_mij, lpl_j_mij = tup
        if i == j:
            raise ValueError("Par (i, i) invalido: diagonal e sempre 0.")
        key = (min(i, j), max(i, j))
        if key in seen:
            warnings.warn(f"Par duplicado {key} ignorado.")
            continue
        seen.add(key)
        d_ij = lpl_self[i] + lpl_self[j] - lpl_i_mij - lpl_j_mij
        # d(i,j) deveria ser >= 0 (M_i e otimo para i). Um negativo forte sinaliza
        # inconsistencia de score/busca, e nao deve ser silenciosamente zerado.
        if d_ij < -1e-6:
            warnings.warn(f"d({i},{j})={d_ij:.4g} < 0: possivel inconsistencia de "
                          f"score/busca (M_i deveria ser otimo para i).")
        if clip_negative and d_ij < 0:
            d_ij = 0.0
        for a, b in ((i, j), (j, i)):   # simetrico
            rows.append(a); cols.append(b); vals.append(d_ij)
    D = sparse.csr_matrix((vals, (rows, cols)),
                          shape=(n_subjects, n_subjects), dtype=dtype)
    mask = sparse.csr_matrix((np.ones(len(vals), bool), (rows, cols)),
                             shape=(n_subjects, n_subjects))
    if not as_sparse:
        D = D.toarray()
    return D, mask


def complete_sparse_distance(D_sparse, method="shortest_path"):
    """
    Completa uma matriz de distancias esparsa em densa, para alimentar metodos
    que exigem D densa e completa (MDS classico, PCoA, ForceScheme, LAMP).

    method="shortest_path": distancia de grafo (geodesica), estilo Isomap --
        distancias faltantes viram o caminho minimo entre os sujeitos.
    """
    if not sparse.issparse(D_sparse):
        return np.asarray(D_sparse, dtype=float)
    if method == "shortest_path":
        # Nota: zeros EXPLICITOS na esparsa (pares identicos, d=0) sao tratados
        # como aresta de peso 0 pelo csgraph. NAO chamar D_sparse.eliminate_zeros(),
        # senao um par identico viraria "ausente" e a geodesica mudaria.
        Dd = shortest_path(D_sparse, method="D", directed=False)
        if not np.isfinite(Dd).all():
            warnings.warn("Grafo desconexo: pares sem caminho ficam inf. "
                          "Considere avaliar mais pares no MDM.")
            finite_max = Dd[np.isfinite(Dd)].max() if np.isfinite(Dd).any() else 1.0
            Dd[~np.isfinite(Dd)] = finite_max * 2.0
        return Dd
    raise ValueError(f"method desconhecido: {method}")


# ----------------------------------------------------------------------------
# helper de clustering (passo IS -> GS do artigo)
# ----------------------------------------------------------------------------
def cluster_labels(D_dense, n_clusters=3, linkage_method="average"):
    """Cluster hierarquico sobre d(i,j) (dendrograma do artigo, Fig.3)."""
    condensed = squareform(np.asarray(D_dense, dtype=float), checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


# ----------------------------------------------------------------------------
# FUNCAO 2 - plot das varias projecoes + dendrograma
# ----------------------------------------------------------------------------
def plot_projections(D, *, labels=None, n_clusters=3,
                     methods=("mds", "tsne", "umap", "force", "lamp", "pcoa"),
                     show_dendrogram=True, annotate=True,
                     random_state=0, figsize=None):
    """
    Plota varias projecoes 2D da matriz de distancias d(i,j) do MDM, tratando
    matrizes esparsas de forma correta por metodo, e (opcional) o dendrograma.

    Retorna (fig, embeddings) -- Figure e dict {metodo: array (S,2)}.
    """
    import matplotlib.pyplot as plt
    is_sparse = sparse.issparse(D)
    D_dense = complete_sparse_distance(D) if is_sparse else np.asarray(D, float)
    S = D_dense.shape[0]
    if labels is None:
        _, labels = cluster_labels(D_dense, n_clusters=n_clusters)
    labels = np.asarray(labels)
    embeddings = {}

    def _mds():
        from sklearn.manifold import MDS
        return MDS(n_components=2, dissimilarity="precomputed",
                   random_state=random_state, normalized_stress="auto"
                   ).fit_transform(D_dense)

    def _pcoa():
        # PCoA / MDS classico: coordenadas = autovetores * sqrt(autovalores) da
        # Gram duplamente centrada B. NAO usar PCA(B), que da V*lambda e estica
        # cada eixo por um fator extra sqrt(lambda_k), distorcendo as distancias.
        J = np.eye(S) - np.ones((S, S)) / S
        B = -0.5 * J @ (D_dense ** 2) @ J
        w, V = np.linalg.eigh(B)                 # autovalores em ordem crescente
        idx = np.argsort(w)[::-1][:2]            # top 2
        lam = np.clip(w[idx], 0.0, None)         # guarda negativos numericos
        return V[:, idx] * np.sqrt(lam)

    def _tsne():
        from sklearn.manifold import TSNE
        perp = min(30, max(5, (S - 1) // 3))
        est = TSNE(n_components=2, metric="precomputed", init="random",
                   perplexity=perp, random_state=random_state)
        try:                                    # esparsa exige vizinhos suficientes
            return est.fit_transform(D if is_sparse else D_dense)
        except Exception as e:
            warnings.warn(f"t-SNE: entrada esparsa falhou ({e}); "
                          f"usando matriz densa completada (geodesica).")
            return est.fit_transform(D_dense)   # fallback denso (geodesico)

    def _umap():
        from umap import UMAP
        est = UMAP(n_components=2, metric="precomputed", init="random",
                   random_state=random_state)
        try:
            return est.fit_transform(D if is_sparse else D_dense)
        except Exception as e:
            warnings.warn(f"UMAP: entrada esparsa falhou ({e}); "
                          f"usando matriz densa completada (geodesica).")
            return est.fit_transform(D_dense)   # fallback denso (geodesico)

    def _force():
        import mppy
        return mppy.force_2d(D_dense)

    def _lamp():
        import mppy
        return mppy.lamp_2d(D_dense)

    runners = {"mds": _mds, "pcoa": _pcoa, "tsne": _tsne,
               "umap": _umap, "force": _force, "lamp": _lamp}
    for m in methods:
        if m not in runners:
            warnings.warn(f"Metodo desconhecido ignorado: {m}")
            continue
        try:
            embeddings[m] = np.asarray(runners[m]())
        except ImportError as e:
            warnings.warn(f"'{m}' pulado (dependencia ausente: {e.name}).")
        except Exception as e:                  # nao derruba os demais paineis
            warnings.warn(f"'{m}' falhou: {e}")

    n_panels = len(embeddings) + (1 if show_dendrogram else 0)
    ncols = min(3, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    k = 0
    if show_dendrogram:
        Z, _ = cluster_labels(D_dense, n_clusters=n_clusters)
        dendrogram(Z, ax=axes[k],
                   labels=[str(i + 1) for i in range(S)],
                   color_threshold=Z[-(n_clusters - 1), 2] if n_clusters > 1 else None)
        axes[k].set_title("Dendrograma (logBF)")
        k += 1
    for m, emb in embeddings.items():
        ax = axes[k]; k += 1
        ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=60,
                   edgecolor="k", linewidth=0.3)
        if annotate:
            for i in range(S):
                ax.annotate(str(i + 1), (emb[i, 0], emb[i, 1]),
                            fontsize=8, ha="center", va="center")
        ax.set_title(m.upper())
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig, embeddings


# ----------------------------------------------------------------------------
# Exemplo minimo reproduzivel (dados sinteticos no lugar das LPLs do MDM)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S = 30                                  # 3 subgrupos (DAG1/DAG2/DAG3)
    grupo = np.repeat([0, 1, 2], S // 3)
    lpl_self = rng.normal(-50, 1, S)
    pairs = []
    for i in range(S):
        for j in range(i + 1, S):
            base = 0.9 if grupo[i] == grupo[j] else 15.0
            d = abs(rng.normal(base, 0.3))
            lpl_i_mij = lpl_self[i] - d / 2   # decompoe d de volta em LPLs
            lpl_j_mij = lpl_self[j] - d / 2
            if rng.random() < 0.4:            # esparsidade: ~40% dos pares
                pairs.append((i, j, lpl_i_mij, lpl_j_mij))
    D, mask = sparse_mdm_dissimilarity(S, lpl_self, pairs, as_sparse=True)
    print("nnz (pares avaliados):", D.nnz // 2, "de", S * (S - 1) // 2)
    fig, emb = plot_projections(D, n_clusters=3)
    fig.savefig("mdm_projections.png", dpi=130)
    print("salvo: mdm_projections.png")
