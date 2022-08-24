# -*- coding: utf-8 -*-
#  File: Home.py
#  Project: 'st-airbnb'
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
# st.markdown("---")

st.write(
    "- **Comp100** - анализ данных по 100 наиболее схожим домам на основе данных airdna.com"
)
st.write("- **Villa** - статистики по дому")
