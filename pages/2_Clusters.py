# -*- coding: utf-8 -*-
#  File: 2_Clusters.py
#  Project: 'analysis-ab'
#  Created by Gennady Matveev (gm@og.ly) on 27-08-2022.
#  Copyright 2022. All rights reserved.
#%%

# Load python libraries for data handling and plotting
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score
import lightgbm as lgb
import shap
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

#%%
# Configure streamlit page
st.set_page_config(
    page_title="Clusters",
    page_icon="./images/nazca_lines.ico",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)

st.write("#### Кластерный анализ. Целевая переменная - годовая выручка.")
#%%
# Load the data set with houses

with open("./data/df2_comp100.pkl", "rb") as f:
    df = pickle.load(f)
# Show the first lines of the data set to get an idea what's in there.
# df.head()
#%%
df.drop(columns=["name", "url", "adr", "occupancy", "sim_score"], inplace=True)
# df.shape

#%%
# Define the standard X (feature matrix) and target series y (not used in here)
X = df.drop(columns=["revenue"])  # adr
all_features = X.columns
y = df["revenue"]  # adr

st.write("Данные - базовая статистика")
st.dataframe(df.describe())
#%%
# Scale features

# scaler = StandardScaler()
scaler = (
    RobustScaler()
)  # Take the robust scaler when data contains outliers that you want to remove

X_scaled = scaler.fit_transform(X)

st.markdown("---")

st.write("##### PCA (Анализ главных компонент)")

pca = PCA().fit(X_scaled)

st.write("Состав двух первых главных компонент")

twod_pca = PCA(n_components=2)
X_pca = twod_pca.fit_transform(X_scaled)

# Add first two PCA components to the dataset
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

df["member"] = 1
# df.groupby('adr')['member'].transform('count').div(df.shape[0])
df.groupby("revenue")["member"].transform("count").div(df.shape[0])
# selection = alt.selection_multi(fields=['adr'], bind='legend')
# df['adr_weight'] = df['adr'].map(df['adr'].value_counts(normalize=True).to_dict())
selection = alt.selection_multi(fields=["revenue"], bind="legend")
df["revenue_weight"] = df["revenue"].map(
    df["revenue"].value_counts(normalize=True).to_dict()
)

# Draw 20% stratified sample
# alt.Chart(df.sample(20, weights='adr_weight')).mark_circle(size=60).encode(
alt.Chart(df.sample(20, weights="revenue_weight")).mark_circle(size=60).encode(
    x=alt.X("pca1", title="First component"),
    y=alt.Y("pca2", title="Second component"),
    color=alt.Color("adr:N"),
    # tooltip=['adr'],
    tooltip=["revenue"],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
).properties(title="PCA analyse", width=600, height=400).add_selection(selection)

df_twod_pca = pd.DataFrame(
    data=twod_pca.components_.T, columns=["pca1", "pca2"], index=X.columns
)

pca1 = (
    alt.Chart(df_twod_pca.reset_index())
    .mark_bar()
    .encode(
        y=alt.Y("index:O", title=None),
        x="pca1",
        color=alt.Color("pca1", scale=alt.Scale(scheme="viridis")),
        tooltip=[
            alt.Tooltip("index", title="Feature"),
            alt.Tooltip("pca1", format=".2f"),
        ],
    )
)
pca2 = (
    alt.Chart(df_twod_pca.reset_index())
    .mark_bar()
    .encode(
        y=alt.Y("index:O", title=None),
        x="pca2",
        color=alt.Color("pca2", scale=alt.Scale(scheme="viridis")),
        tooltip=[
            alt.Tooltip("index", title="Feature"),
            alt.Tooltip("pca2", format=".2f"),
        ],
    )
)

pca_plot = (pca1 & pca2).properties(
    title="Loadings of the first two principal components"
)
st.altair_chart(pca_plot)
#%%
# Find optimal number of clusters

# km_scores= []
# km_silhouette = []
km_db_score = []
for i in range(2, X.shape[1]):
    km = KMeans(n_clusters=i, random_state=42).fit(X_scaled)
    preds = km.predict(X_scaled)

    # print(f'Score for number of cluster(s) {i}: {km.score(X_scaled):.3f}')
    # km_scores.append(-km.score(X_scaled))

    # silhouette = silhouette_score(X_scaled,preds)
    # km_silhouette.append(silhouette)
    # print(f'Silhouette score for number of cluster(s) {i}: {silhouette:.3f}')

    db = davies_bouldin_score(X_scaled, preds)
    km_db_score.append(db)
    # print(f'Davies Bouldin score for number of cluster(s) {i}: {db:.3f}')

    # print('-'*100)
#%%
st.write("Определение оптимального количества кластеров: N -> 3")
df_plot = pd.DataFrame(
    {
        "Number of clusters": [i for i in range(2, X.shape[1])],
        "davies bouldin score": km_db_score,
    }
)
db_score = (
    alt.Chart(df_plot)
    .mark_line(point=True)
    .encode(
        x=alt.X("Number of clusters:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("davies bouldin score", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("davies bouldin score", format=".2f")],
    )
    .properties(
        title="Davies Bouldin score ifo number of cluster", width=600, height=250
    )
)
st.altair_chart(db_score)
#%%

X = X.values

#%%

# Make three clusters

st.markdown("---")


km = KMeans(n_clusters=3, random_state=42).fit(X_scaled)  # Was 3
preds = km.predict(X_scaled)
pd.Series(preds).value_counts()  # How many houses are in each cluster ?

df_km = pd.DataFrame(data={"pca1": X_pca[:, 0], "pca2": X_pca[:, 1], "cluster": preds})

# Add scaled data with the input features

for i, c in enumerate(all_features):
    # Reverse scaling
    df_km[c] = X[:, i]  # X_scaled[:,i]

#%%
st.markdown("**Кластеры и их характеристики**")

brush = alt.selection(type="interval")

domain = [0, 1, 2]  # , 3]
range_ = ["red", "darkblue", "green"]  # , 'magenta']

points = (
    alt.Chart(df_km)
    .mark_circle(size=60)
    .encode(
        x="pca1",
        y="pca2",
        color=alt.condition(brush, "cluster:N", alt.value("lightgray")),
        tooltip=list(all_features),
    )
    .properties(width=300, height=300)
    .add_selection(brush)
)

boxplots = alt.vconcat()
for measure in all_features:
    boxplot = (
        alt.Chart(df_km)
        .mark_boxplot()
        .encode(
            x=alt.X(measure, axis=alt.Axis(titleX=470, titleY=0)),
        )
        .transform_filter(brush)
    )
    boxplots &= boxplot

chart = alt.hconcat(points, boxplots)

st.altair_chart(chart)
#%%
st.markdown("---")
st.write("**Важность признаков - SHAP анализ**")


def plot_cluster(df, selected_columns, clusternr):
    points = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("pca1", title="Principal component 1 (pca1)"),
            y=alt.Y("pca2", title="Principal component 2 (pca2)"),
            color=alt.condition(
                alt.FieldEqualPredicate(field="cluster", equal=clusternr),
                "cluster:N",
                alt.value("lightgray"),
            ),
            tooltip=list(all_features) + ["cluster"],
        )
        .properties(width=300, height=300)
    )

    boxplots = alt.vconcat()
    for measure in [c for c in selected_columns]:
        boxplot = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X(measure, axis=alt.Axis(titleX=480, titleY=0)),
            )
            .transform_filter(alt.FieldEqualPredicate(field="cluster", equal=clusternr))
        )
        boxplots &= boxplot
    return points, boxplots


points, boxplots = plot_cluster(df_km, all_features, 0)
c0 = alt.hconcat(points, boxplots).properties(title="Cluster 0")
points, boxplots = plot_cluster(df_km, all_features, 1)
c1 = alt.hconcat(points, boxplots).properties(title="Cluster 1")
points, boxplots = plot_cluster(df_km, all_features, 2)
c2 = alt.hconcat(points, boxplots).properties(title="Cluster 2")
# points, boxplots = plot_cluster(df_km, all_features, 3)
# c3 = alt.hconcat(points, boxplots).properties(title='Cluster 3')

st.altair_chart(alt.vconcat(c0, c1, c2))  # , c3))
#%%
params = lgb.LGBMClassifier().get_params()

params["objective"] = "multiclass"  # the target to predict is the number of the cluster
params["is_unbalance"] = True
params["n_jobs"] = -1
params["random_state"] = 42

mdl = lgb.LGBMClassifier(**params)


@st.cache
def lgbm_predict(X):
    X = df.drop(columns=["revenue", "pca1", "pca2", "member"])  # adr
    mdl.fit(X, preds)
    return mdl.predict_proba(X)


y_pred = lgbm_predict(X)
#%%
# SHAP explainer
explainer = shap.TreeExplainer(mdl)


@st.cache
def get_shap_values(X):
    return explainer.shap_values(X)


shap_values = get_shap_values(X)
#%%
fig, ax = plt.subplots()
plt.title("Feature importance based on SHAP values")
shap.summary_plot(shap_values, X)
st.pyplot(fig)  # , bbox_inches='tight')

#%%
st.markdown("---")
st.write("##### Кластеры и их характеристики с точки зрения важности признаков")
st.write("**Кластер 0**")


cnr = 0
feature_order = np.argsort(np.sum(np.abs(shap_values[cnr]), axis=0))
points, boxplots = plot_cluster(
    df_km, [X.columns[i] for i in feature_order][::-1][:6], 0
)
c0 = alt.hconcat(points, boxplots).properties(title="Cluster 0")
st.altair_chart(c0)  # .show()
#%%
cnr = 0

fig, ax = plt.subplots()
shap.summary_plot(shap_values[cnr], X, max_display=30)  # , show=False)
plt.title(f"Cluster {cnr}")
st.pyplot(plt)

st.write("**Характеристики кластера 0:**")
st.markdown(
    """ **Меньшие по размеру дома**  
            - меньшее количество кроватей  
            - меньшее количество гостей  
            - меньшее количество ванных
            - скорее высокий рейтинг"""
)

#%%
st.markdown("---")
st.write("**Кластер 1**")

print(X.columns)

cnr = 1
feature_order = np.argsort(np.sum(np.abs(shap_values[cnr]), axis=0))
points, boxplots = plot_cluster(
    df_km, [X.columns[i] for i in feature_order][::-1][:5], cnr
)
c1 = alt.hconcat(points, boxplots).properties(title=f"Cluster {cnr}")
c1  # .show()
#%%
cnr = 1
ig, ax = plt.subplots()
shap.summary_plot(shap_values[cnr], X, max_display=30)  # , show=False)
plt.title(f"Cluster {cnr}")
st.pyplot(plt)

st.write("**Характеристики кластера 1:**")
st.markdown(
    """ **Дома - ветераны**  
            - большое количество отзывов  
            - более удаленные от города  
            - высокий рейтинг"""
)
#%%
st.markdown("---")
st.write("**Кластер 2**")

cnr = 2
feature_order = np.argsort(np.sum(np.abs(shap_values[cnr]), axis=0))
points, boxplots = plot_cluster(
    df_km, [X.columns[i] for i in feature_order][::-1][:6], cnr
)
c2 = alt.hconcat(points, boxplots).properties(title=f"Cluster {cnr}")
c2  # .show()
#%%
cnr = 2
ig, ax = plt.subplots()
shap.summary_plot(shap_values[cnr], X, max_display=30)  # , show=False)
plt.title(f"Cluster {cnr}")
st.pyplot(plt)

st.write("**Характеристики кластера 2:**")
st.markdown(
    """**Большие дома**  
            - большое количество гостей  
            - большое количество ванных  
            - среднее удаление от города
            - любой рейтинг"""
)
#%%
st.markdown("---")

df_km["revenue"] = y
st.write("**Статистика годовой выручки по кластерам**")
st.dataframe(df_km.groupby("cluster")["revenue"].describe())
#%%
# Let's plot the distributions of adr score per cluster
st.write("**Распределение годовой выручки по кластерам**")

adr_chart = (
    alt.Chart(df_km)
    .transform_density(
        # density='adr',
        density="revenue",
        # bandwidth=10,
        groupby=["cluster"],
        extent=[0, 130000],
        counts=True,
        steps=50,
    )
    .mark_area(opacity=0.75)
    .encode(  # area
        alt.X("value:Q"),
        alt.Y("density:Q", stack=None),  #'zero'
        alt.Color("cluster:N"),
        tooltip=["cluster:N"],
    )
    .properties(title="Distribution of revenue per cluster", width=800, height=400)
)
adr_chart
