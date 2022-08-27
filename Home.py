# -*- coding: utf-8 -*-
#  File: Home.py
#  Project: 'analysis-ab'
#  Created by Gennady Matveev (gm@og.ly) on 22-08-2022.
#  Copyright 2022. All rights reserved.
#%%
# Import libraries

import pandas as pd
import altair as alt
import streamlit as st

#%%
st.set_page_config(
    page_title="Home",
    page_icon="./images/nazca_lines.ico",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)
from PIL import Image

image = Image.open("./images/header.png")
st.image(image)
st.write("#### Airbnb - анализ")

st.write(
    "**Comp100** - анализ данных по 100 наиболее схожим домам на основе данных airdna.com")

st.write(
    "**Clusters** - кластерный анализ Comp100"
)

st.markdown("""
    Обозначения:  
    - beds - количество кроватей  
    - bathrooms - ванные  
    - guests - максимальное количество гостей  
    - rating - рейтинг  
    - reviews - количество отзывов  
    - distance - расстояние до центра Лукки  
    - occupancy - заполняемость  
    - adr - средняя дневная ставка аренды  
    - revenue - годовая выручка  
    - sim_score - степень похожести"""
)
st.write("**Villa** - статистики по дому")
