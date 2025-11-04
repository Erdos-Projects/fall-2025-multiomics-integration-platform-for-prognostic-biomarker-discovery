#Upload files from PC (interactive button)
from google.colab import files
import pandas as pd

print("Please choose these 3 files from your computer:")
print("  1. data_rppa.txt")
print("  2. data_mrna_seq_v2_rsem.txt")
print("  3. The_EGFRvIII_Transcriptome_in_glioblastoma_data_public_Zenodo.csv")

uploaded = files.upload()

# Load each file into pandas, keeping it simple for now
rppa = None
rna_raw = None
egfrviii = None

for fn in uploaded.keys():
    if "rppa" in fn.lower():
        rppa = pd.read_csv(fn, sep="\t", index_col=0)
        print("Loaded RPPA:", rppa.shape)
    elif "rna" in fn.lower():
        rna_raw = pd.read_csv(fn, sep="\t", low_memory=False)
        print("Loaded RNA raw:", rna_raw.shape)
    elif "egfr" in fn.lower():
        egfrviii = pd.read_csv(fn)
        print("Loaded EGFRvIII:", egfrviii.shape)

print("\nHeads of the three dataframes:")
print("\nRPPA head (5x5):")
display(rppa.iloc[:5, :5])

print("\nRNA raw head (5 rows):")
display(rna_raw.head())

print("\nEGFRvIII head (5 rows):")
display(egfrviii.head())


#RNA (samples × genes) and compute RPPA↔RNA overlap

import re
import pandas as pd
import numpy as np

# 1) choose the gene column (prefer Hugo_Symbol if present)
gene_col = None
for c in rna_raw.columns:
    if str(c).lower() in ("hugo_symbol", "gene", "gene_name", "symbol", "genes"):
        gene_col = c
        break
if gene_col is None:
    gene_col = rna_raw.columns[0]  # fallback

# 2) detect TCGA sample columns (TCGA-XX-XXXX-YY...)
tcga_pat = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", re.IGNORECASE)
sample_cols = [c for c in rna_raw.columns if tcga_pat.match(str(c))]

# 3) slim to gene + sample cols, drop NA genes, group duplicate genes by mean
rna_slim = rna_raw[[gene_col] + sample_cols].copy()
rna_slim = rna_slim.dropna(subset=[gene_col])
rna_slim = rna_slim.groupby(gene_col, as_index=True).mean(numeric_only=True)

# 4) transpose to samples × genes
rna = rna_slim.T
rna.index.name = "Sample_ID"

# 5) harmonize RPPA columns and compute intersection
rppa.columns = pd.Index([c.strip() for c in rppa.columns])
common_samples = sorted(set(rppa.columns) & set(rna.index))

print("Parsed RNA matrix (samples × genes):", rna.shape)
print("RPPA matrix (features × samples):  ", rppa.shape)
print("Common samples:", len(common_samples))
print("First 10 common samples:", common_samples[:10])

# 6) quick QC counts (before any normalization)
rna_na_total = int(rna.isna().sum().sum())
rppa_na_rows = int((rppa.isna().sum(axis=1) > 0).sum())
rna_zero_var = int((rna.std(axis=0) == 0).sum())
rppa_zero_var = int((rppa.std(axis=1) == 0).sum())

print("\nQC:")
print("  RNA total NA cells:", rna_na_total)
print("  RPPA rows with any NA:", rppa_na_rows)
print("  RNA genes with zero variance:", rna_zero_var)
print("  RPPA features with zero variance:", rppa_zero_var)

# 7) show a tiny peek so we can visually confirm structure
print("\nRNA (5 samples × 5 genes):")
display(rna.iloc[:5, :5])

print("\nRPPA (5 features × 5 samples):")
display(rppa.iloc[:5, :5])


#Normalize (z-score) and clean

import numpy as np

# RPPA: z-score each feature (row)
rppa_z = rppa.sub(rppa.mean(axis=1), axis=0)
rppa_z = rppa_z.div(rppa.std(axis=1).replace(0, np.nan), axis=0)

# RNA: z-score each gene (column)
rna_z = (rna - rna.mean(axis=0)) / rna.std(axis=0).replace(0, np.nan)

# Subset to the 77 common samples
rppa_z = rppa_z[common_samples]
rna_z  = rna_z.loc[common_samples]

print("RPPA_z:", rppa_z.shape)
print("RNA_z :", rna_z.shape)

print("\nMissing values check:")
print("  RPPA_z any NaN:", rppa_z.isna().any().any())
print("  RNA_z any NaN :", rna_z.isna().any().any())


#Strict numeric clean, impute, drop-zero-var, z-score, PCA (RPPA)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Start from RPPA restricted to common samples
R = rppa.loc[:, common_samples].copy()

# 1) Force numeric (coerce non-numeric to NaN)
R = R.apply(pd.to_numeric, errors="coerce")

# 2) Drop rows that are entirely NaN
n_allna_before = int((R.isna().all(axis=1)).sum())
R = R.loc[~R.isna().all(axis=1)]
print("Rows dropped (all-NaN):", n_allna_before)

# 3) Impute remaining NaNs per row with the row median
R_imp = R.apply(lambda row: row.fillna(row.median()), axis=1)

# 4) Drop rows that are constant after imputation (std == 0)
row_std = R_imp.std(axis=1)
n_const = int((row_std == 0).sum())
R_imp = R_imp.loc[row_std > 0]
print("Rows dropped (zero-variance):", n_const)

# 5) Z-score per row
R_z = R_imp.sub(R_imp.mean(axis=1), axis=0)
R_z = R_z.div(R_imp.std(axis=1), axis=0)

# 6) Final guard: replace any residual NaN with 0
had_nan = bool(R_z.isna().any().any())
R_z = R_z.fillna(0.0)
print("Had NaN after z-score (pre-fill)?", had_nan)
print("Any NaN now?:", bool(R_z.isna().any().any()))

# 7) PCA (samples x features)
X = R_z.T.values
pca_rppa = PCA(n_components=5, random_state=0).fit(X)
scores = pca_rppa.transform(X)

# Scree
plt.figure(figsize=(5,3))
plt.plot(range(1,6), pca_rppa.explained_variance_ratio_, marker='o')
plt.xlabel("PC")
plt.ylabel("Variance explained")
plt.title("RPPA PCA — Scree")
plt.show()

# PC1 vs PC2
plt.figure(figsize=(5,4))
plt.scatter(scores[:,0], scores[:,1], alpha=0.7)
plt.xlabel(f"PC1 ({pca_rppa.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_rppa.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("RPPA PCA — samples")
plt.show()

# Loadings for PC1 (feature contributions)
pc1_load = pd.Series(pca_rppa.components_[0], index=R_z.index).sort_values(ascending=False)
print("Top 5 PC1 loadings:")
print(pc1_load.head(5))
print("\nBottom 5 PC1 loadings:")
print(pc1_load.tail(5))


#PCA on RNA (restrict to high-variance genes)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Subset to common samples
rna_sub = rna.loc[common_samples]

# Compute variance per gene
gene_var = rna_sub.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(2000).index
rna_hv = rna_sub[top_genes]

# Z-score each gene (re-normalize within this subset)
rna_hv_z = (rna_hv - rna_hv.mean(axis=0)) / rna_hv.std(axis=0).replace(0, np.nan)
rna_hv_z = rna_hv_z.fillna(0.0)

print("RNA high-variance subset:", rna_hv_z.shape)

# PCA
X_rna = rna_hv_z.values
pca_rna = PCA(n_components=5, random_state=0).fit(X_rna)
rna_embed = pca_rna.transform(X_rna)

# Scree
plt.figure(figsize=(5,3))
plt.plot(range(1,6), pca_rna.explained_variance_ratio_, marker='o')
plt.xlabel("PC")
plt.ylabel("Variance explained")
plt.title("RNA PCA — Scree")
plt.show()

# PC1 vs PC2 scatter
plt.figure(figsize=(5,4))
plt.scatter(rna_embed[:,0], rna_embed[:,1], alpha=0.7)
plt.xlabel(f"PC1 ({pca_rna.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_rna.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("RNA PCA — samples (top 2000 variable genes)")
plt.show()

# Top/bottom loadings for PC1
pc1_load_rna = pd.Series(pca_rna.components_[0], index=rna_hv_z.columns).sort_values(ascending=False)
print("Top 5 PC1 loadings (RNA):")
print(pc1_load_rna.head(5))
print("\nBottom 5 PC1 loadings (RNA):")
print(pc1_load_rna.tail(5))


#Side-by-side PCA plots for RPPA vs RNA

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10,4))

# RPPA scatter
axes[0].scatter(scores[:,0], scores[:,1], alpha=0.7)
axes[0].set_xlabel(f"PC1 ({pca_rppa.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca_rppa.explained_variance_ratio_[1]*100:.1f}% var)")
axes[0].set_title("RPPA PCA (proteins/phospho)")

# RNA scatter
axes[1].scatter(rna_embed[:,0], rna_embed[:,1], alpha=0.7, color="darkorange")
axes[1].set_xlabel(f"PC1 ({pca_rna.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca_rna.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].set_title("RNA PCA (top 2000 variable genes)")

plt.tight_layout()
plt.show()


#Compare sample distances across RNA vs RPPA

from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

# Use PCA scores (PC1–PC5) for RNA and RPPA
rna_dist = pairwise_distances(rna_embed[:, :5])   # 77×77
rppa_dist = pairwise_distances(scores[:, :5])     # 77×77

# Flatten upper triangles for correlation
mask = np.triu(np.ones(rna_dist.shape), k=1).astype(bool)
rna_flat = rna_dist[mask]
rppa_flat = rppa_dist[mask]

# Correlation between distance structures
from scipy.stats import spearmanr
corr, pval = spearmanr(rna_flat, rppa_flat)

print(f"RNA–RPPA sample distance correlation: r={corr:.3f}, p={pval:.2e}")

# Scatterplot of distances
plt.figure(figsize=(5,5))
plt.scatter(rna_flat, rppa_flat, alpha=0.3)
plt.xlabel("RNA sample distance (PC1–5)")
plt.ylabel("RPPA sample distance (PC1–5)")
plt.title(f"Sample-level concordance RNA vs RPPA (Spearman r={corr:.2f})")
plt.show()

# Heatmap comparison
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.heatmap(rna_dist, cmap="viridis", ax=axes[0])
axes[0].set_title("RNA sample distances (PC1–5)")
sns.heatmap(rppa_dist, cmap="viridis", ax=axes[1])
axes[1].set_title("RPPA sample distances (PC1–5)")
plt.show()


#Feature-level correlation (RNA vs RPPA)

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Match features: extract gene names from RPPA (before the '|')
rppa_genes = rppa.index.to_series().str.split("|").str[0]
common_genes = rppa_genes[rppa_genes.isin(rna.columns)].unique()

results = []
for g in common_genes:
    # RNA vector
    rna_vec = rna_z[g].loc[common_samples]
    # All RPPA rows mapping to this gene
    rows = rppa.index[rppa.index.str.startswith(g+"|")]
    for row in rows:
        rppa_vec = rppa_z.loc[row, common_samples]
        if rna_vec.std() > 0 and rppa_vec.std() > 0:
            r, p = pearsonr(rna_vec, rppa_vec)
            results.append((g, row, r, p, len(common_samples)))

res_df = pd.DataFrame(results, columns=["Gene","RPPA_feature","PearsonR","Pval","N"])
res_df["log10P"] = -np.log10(res_df["Pval"])

print("Correlated features:", res_df.shape)
print(res_df.sort_values("PearsonR", ascending=False).head(10))

# Volcano plot
plt.figure(figsize=(6,5))
plt.scatter(res_df["PearsonR"], res_df["log10P"], alpha=0.6)
plt.axvline(0, color="grey", linestyle="--")
plt.xlabel("Pearson correlation (RNA vs RPPA)")
plt.ylabel("-log10(p-value)")
plt.title("RNA–RPPA feature-level correlations")
plt.show()


#Heatmaps of top concordant and discordant features

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sort by correlation
res_sorted = res_df.sort_values("PearsonR", ascending=False)

# Top 10 concordant
top10 = res_sorted.head(10).copy()
# Bottom 10 discordant (lowest correlations)
bot10 = res_sorted.tail(10).copy()

def _prep_rna_mat(genes):
    # keep only genes that exist in rna_z
    genes = [g for g in genes if g in rna_z.columns]
    M = rna_z.loc[common_samples, genes]          # samples × genes
    # drop duplicate columns if any
    M = M.loc[:, ~M.columns.duplicated()]
    # drop columns/rows that are all NaN or zero variance
    keep_cols = (M.notna().any(axis=0)) & (M.std(axis=0, ddof=1) > 0)
    M = M.loc[:, keep_cols]
    keep_rows = (M.notna().any(axis=1)) & (M.std(axis=1, ddof=1) > 0)
    M = M.loc[keep_rows]
    return M

def _prep_rppa_mat(features):
    feats = [f for f in features if f in rppa_z.index]
    M = rppa_z.loc[feats, common_samples].T        # samples × features
    # drop duplicate columns if any
    M = M.loc[:, ~M.columns.duplicated()]
    # drop columns/rows that are all NaN or zero variance
    keep_cols = (M.notna().any(axis=0)) & (M.std(axis=0, ddof=1) > 0)
    M = M.loc[:, keep_cols]
    keep_rows = (M.notna().any(axis=1)) & (M.std(axis=1, ddof=1) > 0)
    M = M.loc[keep_rows]
    return M

def plot_heatmap_pair(subset, title_prefix, tag):
    genes = subset["Gene"].tolist()
    feats = subset["RPPA_feature"].tolist()

    M_rna  = _prep_rna_mat(genes)
    M_rppa = _prep_rppa_mat(feats)

    # If a matrix ends up empty, report and skip plotting to avoid blank figs
    if M_rna.shape[1] == 0 or M_rna.shape[0] == 0:
        print(f"[WARN] RNA matrix is empty for {title_prefix} (after cleaning).")
    else:
        g1 = sns.clustermap(
            M_rna, cmap="vlag", center=0, robust=True,
            figsize=(7,7), dendrogram_ratio=(.15,.15),
            cbar_pos=(0.02, .8, .03, .15)
        )
        g1.ax_heatmap.set_title(f"{title_prefix} (RNA)", pad=12)
        plt.savefig(f"heatmap_{tag}_RNA.png", dpi=150, bbox_inches="tight")
        plt.show()

    if M_rppa.shape[1] == 0 or M_rppa.shape[0] == 0:
        print(f"[WARN] RPPA matrix is empty for {title_prefix} (after cleaning).")
    else:
        g2 = sns.clustermap(
            M_rppa, cmap="vlag", center=0, robust=True,
            figsize=(7,7), dendrogram_ratio=(.15,.15),
            cbar_pos=(0.02, .8, .03, .15)
        )
        g2.ax_heatmap.set_title(f"{title_prefix} (RPPA)", pad=12)
        plt.savefig(f"heatmap_{tag}_RPPA.png", dpi=150, bbox_inches="tight")
        plt.show()

    # Return shapes so you can sanity-check
    print(f"{title_prefix} shapes — RNA {M_rna.shape}, RPPA {M_rppa.shape}")

print("Top 10 correlated genes/features:")
display(top10[["Gene","RPPA_feature","PearsonR"]])
plot_heatmap_pair(top10, "Top concordant features", "top_concordant")

print("Bottom 10 correlated genes/features:")
display(bot10[["Gene","RPPA_feature","PearsonR"]])
plot_heatmap_pair(bot10, "Bottom discordant features", "bottom_discordant")


#Match RPPA features to RNA genes by symbol
rppa_genes = [f.split("|")[0] for f in rppa.index]  # take leftmost part before '|'
rna_genes = set(rna_z.columns)

# Keep only those RPPA features that have RNA match
matched = [(g, f) for g, f in zip(rppa_genes, rppa.index) if g in rna_genes]

matched_df = pd.DataFrame(matched, columns=["Gene", "RPPA_feature"])
print("Matched gene–protein features:", matched_df.shape)
print(matched_df.head(20))

# STEP 2 — Subset RNA and RPPA to matched features
rna_matched = rna_z[matched_df["Gene"]]
rppa_matched = rppa_z.loc[matched_df["RPPA_feature"]].T

print("RNA matched:", rna_matched.shape)
print("RPPA matched:", rppa_matched.shape)

# STEP 3 — Save for pathway analysis
matched_df.to_csv("matched_features.csv", index=False)


import gseapy as gp
libs = gp.get_library_name()  # list all supported libraries
print(libs[:50])  # just print first 50 to check


import gseapy as gp
import pandas as pd

# Use your matched genes
gene_list = matched_df["Gene"].tolist()

# Run enrichment against BioPlanet (human pathways)
enr = gp.enrichr(
    gene_list=gene_list,
    gene_sets=["BioPlanet_2019"],   # pathway database
    organism="Human",
    outdir=None,
    cutoff=0.5
)

# Sort and show top hits
res = enr.results.sort_values("Adjusted P-value").head(20)
print(res[["Term", "Overlap", "Adjusted P-value", "Genes"]])


#Bubble plot of enrichment results
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use your enrichment results dataframe (enr.results)
df = enr.results.copy()

# Sort by adjusted p-value and keep top 20
df_top = df.sort_values("Adjusted P-value").head(20)

# Make bubble plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_top,
    x=-np.log10(df_top["Adjusted P-value"]),
    y="Term",
    size=df_top["Overlap"].apply(lambda x: int(x.split("/")[0])),  # gene hits
    hue=-np.log10(df_top["Adjusted P-value"]),
    palette="viridis",
    sizes=(50, 400),
    legend="full"
)

plt.title("Top 20 Enriched BioPlanet Pathways", fontsize=14)
plt.xlabel("-log10(Adjusted P-value)")
plt.ylabel("")
plt.legend(title="Overlap size", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


!pip install gseapy


import gseapy as gp

# Your matched gene list
gene_list = matched_df["Gene"].tolist()

# Run enrichment against KEGG 2019 Human (or you can try GO_Biological_Process_2021, Reactome_2016, etc.)
enr = gp.enrichr(
    gene_list=gene_list,
    gene_sets=["KEGG_2019_Human"],
    organism='Human',   # or 'hsapiens'
    outdir=None,        # don’t write files, just keep in memory
    cutoff=0.5          # FDR threshold
)

# Inspect results
print(enr.results.head(10))


import matplotlib.pyplot as plt
import seaborn as sns

# Take the enrichment results
df = enr.results.copy()

# Keep top 15 pathways (sorted by Adjusted P-value)
top_df = df.sort_values("Adjusted P-value").head(15)

# Clean up term names for plotting
top_df["Term"] = top_df["Term"].str.replace("Homo sapiens", "", regex=False)
top_df["-log10(FDR)"] = -np.log10(top_df["Adjusted P-value"] + 1e-300)  # avoid -inf

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=top_df,
    x="-log10(FDR)",
    y="Term",
    size=top_df["Overlap"].apply(lambda x: int(x.split("/")[0])),
    hue="-log10(FDR)",
    palette="Reds",
    sizes=(50, 400),
    edgecolor="k"
)

plt.title("KEGG Pathway Enrichment (Matched RNA–RPPA features)", fontsize=14)
plt.xlabel("-log10(FDR adjusted p-value)")
plt.ylabel("")
plt.legend(title="Gene Count", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()


#Pathway overlap network
import networkx as nx
import matplotlib.pyplot as plt

# Use the enrichment results from gseapy
df = enr.results.copy()

# Keep top 15 pathways
top_df = df.head(15)

# Build graph
G = nx.Graph()

# Add nodes with size = overlap count
for _, row in top_df.iterrows():
    term = row["Term"]
    overlap = int(row["Overlap"].split("/")[0])  # extract numerator
    G.add_node(term, size=overlap, pval=row["Adjusted P-value"])

# Add edges if two pathways share ≥5 genes
for i in range(len(top_df)):
    for j in range(i+1, len(top_df)):
        genes_i = set(top_df.iloc[i]["Genes"].split(";"))
        genes_j = set(top_df.iloc[j]["Genes"].split(";"))
        shared = genes_i & genes_j
        if len(shared) >= 5:
            G.add_edge(top_df.iloc[i]["Term"], top_df.iloc[j]["Term"], weight=len(shared))

# Draw network
plt.figure(figsize=(10,8))
pos = nx.spring_layout(G, k=0.4, seed=42)

# Node sizes scaled by overlap
sizes = [G.nodes[n]["size"]*40 for n in G.nodes]
colors = [-np.log10(G.nodes[n]["pval"]) for n in G.nodes]

nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap="Reds")
nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=9)

cbar = plt.colorbar(nodes)
cbar.set_label("-log10(FDR)", fontsize=12)

plt.title("KEGG Pathway Overlap Network (RNA–RPPA matched features)")
plt.axis("off")
plt.show()


#Pathway-level RNA vs RPPA activity heatmap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use top enriched pathways from BioPlanet results
top_pathways = enr.results["Term"].head(10).tolist()

# Build a mapping of genes to pathways (from enrichment results)
pathway_map = {}
for _, row in enr.results.iterrows():
    if row["Term"] in top_pathways:
        genes = row["Genes"].split(";")
        pathway_map[row["Term"]] = [g.strip() for g in genes if g in rna_z.columns]

# Calculate activity score = mean z-score per pathway per sample
rna_pathway = pd.DataFrame(index=common_samples)
rppa_pathway = pd.DataFrame(index=common_samples)

for pw, genes in pathway_map.items():
    if len(genes) > 1:
        rna_pathway[pw] = rna_z[genes].mean(axis=1)
        # RPPA: select matching protein features for those genes
        matched_proteins = matched_df[matched_df["Gene"].isin(genes)]["RPPA_feature"]
        if len(matched_proteins) > 1:
            rppa_pathway[pw] = rppa_z.loc[matched_proteins, common_samples].T.mean(axis=1)

# Merge RNA + RPPA activity
merged_activity = pd.concat([rna_pathway.add_suffix("_RNA"),
                             rppa_pathway.add_suffix("_RPPA")], axis=1)

# Heatmap (samples x pathways)
plt.figure(figsize=(12,6))
sns.clustermap(merged_activity, cmap="vlag", col_cluster=True, row_cluster=True, figsize=(14,10))
plt.suptitle("Pathway-level activity (RNA vs RPPA)", y=1.02)
plt.show()


#RNA–RPPA correlation network within pathways

import networkx as nx
import numpy as np

# Filter correlations from res_df (already computed earlier)
# Keep only strong correlations (PearsonR > 0.5, FDR < 0.05)
strong_corr = res_df[(res_df["PearsonR"] > 0.5) & (res_df["Pval"] < 0.05)]

# Build graph
G = nx.Graph()

for _, row in strong_corr.iterrows():
    gene = row["Gene"]
    prot = row["RPPA_feature"]
    corr = row["PearsonR"]

    # Add nodes
    G.add_node(gene, type="RNA")
    G.add_node(prot, type="RPPA")

    # Add edge with correlation weight
    G.add_edge(gene, prot, weight=corr)

# Layout
pos = nx.spring_layout(G, seed=42, k=0.5)

plt.figure(figsize=(12, 10))

# Draw RNA nodes (circles)
rna_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "RNA"]
nx.draw_networkx_nodes(G, pos, nodelist=rna_nodes, node_color="skyblue", node_size=500, label="RNA")

# Draw RPPA nodes (squares)
prot_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "RPPA"]
nx.draw_networkx_nodes(G, pos, nodelist=prot_nodes, node_color="lightcoral", node_shape="s", node_size=600, label="Protein")

# Draw edges, weighted by correlation
edges = G.edges(data=True)
weights = [d['weight']*2 for (u,v,d) in edges]
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6)

# Labels
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("RNA–RPPA Correlation Network (strong signals)", fontsize=14)
plt.legend(scatterpoints=1)
plt.axis("off")
plt.show()

print("Network stats:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

