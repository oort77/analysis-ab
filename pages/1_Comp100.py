# -*- coding: utf-8 -*-
#  File: 1_comp100.py
#  Project: 'st-airbnb'
#  Created by Gennady Matveev (gm@og.ly) on 22-08-2022.
#  Copyright 2022. All rights reserved.
#%%
# Import libraries

import pandas as pd

# import numpy as np
import altair as alt
import streamlit as st
import gdown
import re

#%%
st.set_page_config(
    page_title="Competition-100",
    page_icon="./images/nazca_lines.ico",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.markdown("#### Analysis of 100 closest competitors")
st.markdown("---")

#%%
# Download and preprocess data
url = "https://drive.google.com/uc?export=download&id=146pWyruVf_MHyjoGEGmzyv3Md5e5cL7o"


@st.experimental_memo
def get_txt():
    txt_file = gdown.download(url, output="./comp100.txt", quiet=True)
    with open(txt_file, "r") as f:
        return f.readlines()


txt = get_txt()
raw_data = list(map(lambda x: x[:-1], txt))
data_dict = {
    i: raw_data[i * 11:(i + 1) * 11]
    for i in range(len(raw_data) // 11)
}

df = pd.DataFrame(data_dict).T
df.columns = [
    "beds",
    "bathrooms",
    "guests",
    "name",
    "rating",
    "reviews",
    "distance",
    "occupancy",
    "adr",
    "revenue",
    "sim_score",
]

int_columns = ["beds", "guests", "reviews", "occupancy", "adr", "sim_score"]
float_columns = ["bathrooms", "rating", "distance", "revenue"]

for i in int_columns:
    df[i] = df[i].apply(lambda x: int(re.findall("\d+", x)[0]))

for i in float_columns:
    df[i] = df[i].apply(lambda x: float(re.findall("\d+\.*\d*", x)[0]))

df["distance"] = df["distance"].apply(lambda x: x * 1000 if x < 30 else x)
df["revenue"] = df["revenue"].apply(lambda x: x * 1000)

cols = df.columns.tolist()
cols = [cols[3]] + cols[:3] + cols[4:]
df = df[cols]

# Get rid of ouliers

outlier_idx = df.loc[df["revenue"] > 200000].index
df2 = df.drop(outlier_idx)

#%%
# Download and append houses URLS
url2 = (
    "https://drive.google.com/uc?export=download&id=1oBqemHT7MQe9w8EITm95OBa98xmDTfOc"
)


@st.experimental_memo
def get_csv():
    csv_file = gdown.download(url2, output="./comp100_urls.csv", quiet=False)
    urls = pd.read_csv(csv_file)
    urls2 = urls.T.reset_index().rename({"index": "url"}, axis=1)
    return pd.concat([df, urls2], axis=1)


df = get_csv()
df["distance"] = df["distance"].astype(int)
df["revenue"] = df["revenue"].astype(int)

# Revenue vs occupancy
# TODO: make it open in another tab

#%%

# Correlations

cor_data = (df.corr().stack().reset_index().rename(columns={
    0: "correlation",
    "level_0": "variable",
    "level_1": "variable2"
}))
cor_data["correlation_label"] = cor_data["correlation"].map("{:.2f}".format)
# cor_data.head(12)

base = alt.Chart(cor_data).encode(x="variable2:O", y="variable:O")

# Text layer with correlation labels
# Colors are for easier readability
text = base.mark_text().encode(
    text="correlation_label",
    color=alt.condition(alt.datum.correlation > 0.5, alt.value("white"),
                        alt.value("black")),
)

# The correlation heatmap itself
cor_plot = (base.mark_rect().encode(color="correlation:Q").properties(
    width=600, height=480, title="Correlations"))

st.altair_chart(cor_plot + text)
#%%
# Interactive crossfilter:
# payouts, occupancy, rating

brush = alt.selection(type="interval", encodings=["x"])

# Define the base chart, with the common parts of the
# background and highlights
base = (alt.Chart().mark_bar(color="steelblue").encode(
    x=alt.X(alt.repeat("column"), type="quantitative",
            bin=alt.Bin(maxbins=20)),
    y="count()",
).properties(width=150, height=200))

# gray background with selection
background = base.encode(color=alt.value("#ddd")).add_selection(brush)

# blue highlights on the transformed data
highlight = base.transform_filter(brush)

# layer the two charts & repeat
crossfilter = (alt.layer(
    background, highlight,
    data=df2).repeat(column=["revenue", "rating", "occupancy"]).properties(
        title="Interactive crossfilter: revenue, rating, occupancy"))

st.altair_chart(crossfilter)
#%%
# Revenue vs occupancy
revenue_vs_occupancy = (alt.Chart(df).transform_filter(
    "datum.revenue < 200000").mark_point().encode(
        alt.X("occupancy:Q", scale=alt.Scale(zero=False)),
        alt.Y("revenue:Q", scale=alt.Scale(zero=False)),
        size="guests:O",
        tooltip=[
            "name",
            "occupancy",
            "revenue",
            "guests",
            "adr",
            "rating",
            "sim_score",
        ],
        color=alt.Color("rating:O", scale=alt.Scale(scheme="category20c")),
        href="url:N",
    ).properties(width=600, height=300, title="Revenue vs Occupancy"))
revenue_vs_occupancy["usermeta"] = {
    "embedOptions": {
        "loader": {
            "target": "_blank"
        }
    }
}

st.altair_chart(revenue_vs_occupancy)
# %%

# Occupancy distribution (rating > 4 vs all)

occupancy_distr = (
    alt.Chart(df).transform_filter(
        "datum.revenue < 200000")  # & datum.guests>10'
    .mark_bar(size=10, color="darkseagreen").encode(
        alt.X("occupancy:Q", bin=True, scale=alt.Scale(zero=False)),
        alt.Y("count()", scale=alt.Scale(zero=False)),
        tooltip=["count()"],
    ).properties(width=600,
                 height=300,
                 title="Occupancy distribution, rating>4?"))

occupancy_distr_rating4plus = (
    alt.Chart(df).transform_filter(
        "datum.revenue < 200000  & datum.rating>4").mark_bar(
            size=10, color="darkorange").encode(
                alt.X("occupancy:Q", bin=True, scale=alt.Scale(zero=False)),
                alt.Y("count()", scale=alt.Scale(zero=False)),
                tooltip=["count()"],
            ).properties(
                width=600,
                height=300,
                # title='Occupancy distribution, airdna_comp100'
            ))
chart1 = occupancy_distr + occupancy_distr_rating4plus

st.altair_chart(chart1)
#%%
# Occupancy vs revenue, is number of reviews important?

slider = alt.binding_range(min=0, max=125, step=2, name="Number of reviews:")
selector = alt.selection_single(name="SelectorName",
                                fields=["cutoff"],
                                bind=slider,
                                init={"cutoff": 0})
occ_vs_rev_by_reviews = (
    alt.Chart(df2).mark_point().encode(
        alt.X("occupancy:Q", scale=alt.Scale(zero=False)),
        alt.Y("revenue:Q", scale=alt.Scale(zero=False)),
        size="rating:O",
        tooltip=["name", "guests", "rating"],
        # href='url:N',
        color=alt.condition(
            alt.datum.reviews >= selector.cutoff,
            alt.value("steelblue"),
            alt.value("darkorange"),
        ),
    ).add_selection(selector).properties(
        width=600,
        height=300,
        title="Occupancy vs revenue: is number of reviews important?",
    ))

st.altair_chart(occ_vs_rev_by_reviews)
# %%

# Revenue distribution (rating > 4 vs all)

outlier_idx = df.loc[df["revenue"] > 200000].index
df2 = df.drop(outlier_idx)

revenue_all = (alt.Chart(df2).transform_calculate(
    color="datum.rating > 0").mark_bar(size=10).encode(
        alt.X("revenue:Q", bin=True, scale=alt.Scale(zero=False)),
        alt.Y("count()", scale=alt.Scale(zero=False)),
        tooltip=["count()"],
    ).properties(width=600,
                 height=300,
                 title="Revenue distribution, rating>4?"))

revenue_4plus = (
    alt.Chart(df2).transform_filter("datum.rating>=4").transform_calculate(
        color="datum.rating > 4")  # ? 'rating>4': 'all ratings'"
    .mark_bar(size=10, opacity=0.5).encode(
        alt.X("revenue:Q", bin=True, scale=alt.Scale(zero=False)),
        alt.Y("count()", scale=alt.Scale(zero=False)),
        color=alt.Color("color:N",
                        scale=alt.Scale(range=["lightgreen", "steelblue"])),
        tooltip=["count()"],
    ))
chart2 = revenue_all + revenue_4plus

st.altair_chart(chart2)

# %%

# Ratings > 4.4 histogram

ratings_histo = (
    alt.Chart(df).transform_filter("datum.rating >= 4.4").mark_bar(
        size=10, color="steelblue")  # & datum.guests>10'
    .encode(
        alt.X("rating:Q", bin=True, scale=alt.Scale(zero=False)),
        alt.Y("count()", scale=alt.Scale(zero=False)),
    ).properties(width=600, height=300, title="Ratings (> 4.4) distribution"))

st.altair_chart(ratings_histo)
# %%
# ADR histogram

adr_histo = (
    alt.Chart(df).mark_bar(
        size=10, color="darkorange")  # & datum.guests>10'
    .encode(
        alt.X("adr:Q", bin=True, scale=alt.Scale(zero=False)),
        alt.Y("count()", 
    ).properties(width=600, height=300, title="Average daily rates distribution"))

st.altair_chart(adr_histo)
#%%

# Visualize median revenue vs rating

revenue_vs_rating = (alt.Chart(df).transform_filter(
    "datum.rating > 4 & datum.revenue<200000").mark_bar(
        size=10, color="darkseagreen").encode(
            alt.X("rating:Q", bin=True, scale=alt.Scale(zero=False)),
            alt.Y("median(revenue):Q", scale=alt.Scale(zero=False)),
        ).properties(width=600,
                     height=300,
                     title="Median revenue vs Rating (> 4)"))
st.altair_chart(revenue_vs_rating)
# %%

# Visualize revenue KDE (rating > 4 vs all)

revenue_kde_4 = (alt.Chart(df).transform_filter(
    "datum.rating > 4 & datum.revenue<200000").transform_density(
        "revenue",
        as_=["revenue",
             "density"]).mark_area(color="darkred", opacity=0.25).encode(
                 y="density:Q",
                 x="revenue:Q",
             ).properties(
                 width=600,
                 height=300,
                 title="Revenue distribution KDE, rating>4 vs all ratings",
             ))

revenue_kde_all = (
    alt.Chart(df).transform_filter("datum.revenue<200000").transform_density(
        "revenue",
        as_=["revenue",
             "density"]).mark_area(color="darkblue", opacity=0.25).encode(
                 y="density:Q",
                 x="revenue:Q",
             ).properties(
                 width=600,
                 height=300,
                 title="Revenue distribution KDE, all ratings",
             ))

final = revenue_kde_4 + revenue_kde_all

st.altair_chart(final)

# %%
st.write("**Source data**")
st.dataframe(df)
