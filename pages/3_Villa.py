# -*- coding: utf-8 -*-
#  File: 3_villa.py
#  Project: 'analysis-ab'
#  Created by Gennady Matveev (gm@og.ly) on 22-08-2022.
#  Copyright 2022. All rights reserved.
#%%
# Import libraries

import pandas as pd
import altair as alt
import streamlit as st
import gdown

#%%
st.set_page_config(
    page_title="Villa",
    page_icon="./images/nazca_lines.ico",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# %%
url = "https://drive.google.com/uc?export=download&id=1IYM-aR0lZ3x1sUSy0v_-QbN8SoqTaaL0"


@st.experimental_memo
def get_csv2():
    return pd.read_csv(
        gdown.download(url, output="./bookings2.csv", quiet=True),
        parse_dates=[3, 4, 5],
        dayfirst=True,
    )


df = get_csv2()
# %%
# Massage dataframe

df["BookedLag"] = (df["Checkin"] - df["Booked"]).apply(lambda x: x.days)
df["Payout"] = df["Payout"].apply(lambda x: float(x.replace(",", "")))
print(df["Guests"])
df["Guest"] = df["Guests"].apply(lambda x: x.split("\n")[0])
df["Adults"] = (
    df["Guests"]
    .apply(lambda x: x.split("\n")[1])
    .apply(lambda x: int(x.split(",")[0].split()[0]))
)
df["Children"] = (
    df["Guests"]
    .apply(lambda x: x.split("\n")[1])
    .apply(lambda x: int(x.split(",")[1].split()[0]) if len(x.split(",")) > 1 else 0)
)
df["Infants"] = (
    df["Guests"]
    .apply(lambda x: x.split("\n")[1])
    .apply(lambda x: int(x.split(",")[2].split()[0]) if len(x.split(",")) > 2 else 0)
)
df["Kids"] = df["Children"] + df["Infants"]
df.drop(columns="Guests", inplace=True)

# %%
# Visualize guests payouts by country
st.write("**Выплаты - хронология**")
guests_by_country = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x=alt.X("Checkin:T", scale=alt.Scale(domain=("2021-07-15", "2022-12-15"))),
        y="Payout:Q",
        color="Country:N",
    )
    .properties(title="Payouts, chronological", width=600, height=300)
    .interactive()
)
guests_by_country.configure_axisX(labelAngle=0)

st.altair_chart(guests_by_country)
# %%
# Visualize Booking lag by country
st.write("**За сколько дней бронировали - по странам**")
booking_lag = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x="Country",
        y="BookedLag",
        color="Country",
        tooltip=["Status", "Booked", "Checkin"],
    )
    .properties(title="Booking lag by country, days", width=600, height=300)
    .interactive()
)

booking_lag.configure_title(fontSize=14, font="Helvetica", anchor="start", color="gray")

mean_by_country = (
    alt.Chart(df)
    .mark_line(color="red", strokeDash=[3, 5], point=True)
    .encode(x="Country", y="mean(BookedLag)")
)

mean_overall = (
    alt.Chart(df)
    .mark_rule(color="green", size=4, strokeDash=[3, 5])
    .encode(y="mean(BookedLag)")
)

annotation = (
    alt.Chart(df)
    .mark_text(
        align="left", baseline="middle", fontSize=12, dx=80, dy=15, color="green"
    )
    .encode(text=alt.Text("mean(BookedLag):N", format=",.1f"))
)

final_plot = (
    booking_lag + mean_by_country + mean_overall + annotation
).configure_axisX(labelAngle=0)

st.altair_chart(final_plot)
# %%
# Visualize Booking lag by kids
st.write("**За сколько дней бронировали - по количеству детей**")
booking_lag_kids = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        alt.X("Kids:O", scale=alt.Scale(zero=False)),
        y="BookedLag",
        color="Country",
        tooltip=["Status", "Booked", "Checkin"],
    )
    .properties(title="Booking lag by number of kids, days", width=600, height=300)
    .interactive()
)

booking_lag.configure_title(fontSize=14, font="Helvetica", anchor="start", color="gray")

mean_by_kids = (
    alt.Chart(df)
    .mark_line(color="red", strokeDash=[3, 5], point=True)
    .encode(x="Kids:O", y="mean(BookedLag)")
)

mean_overall_kids = (
    alt.Chart(df)
    .mark_rule(color="green", size=4, strokeDash=[3, 5])
    .encode(y="mean(BookedLag)")
)

annotation_kids = (
    alt.Chart(df)
    .mark_text(
        align="left", baseline="middle", fontSize=12, dx=10, dy=15, color="green"
    )
    .encode(text=alt.Text("mean(BookedLag):N", format=",.1f"))
)

final_plot = (
    booking_lag_kids + mean_by_kids + mean_overall + annotation_kids
).configure_axisX(labelAngle=0)

st.altair_chart(final_plot)
# %%
# Visualize Payouts by country
st.write("**Размеры выплат по странам**")

payouts = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x="Country",
        y="Payout",
        color="Country",
        tooltip=["Status", "Booked", "Checkin", "Payout"],
    )
    .properties(title="Payouts by country", width=600, height=300)
    .interactive()
)

payouts.configure_title(fontSize=14, font="Helvetica", anchor="start", color="gray")

mean_po_by_country = (
    alt.Chart(df)
    .mark_line(color="orange", strokeDash=[3, 5], point=True)
    .encode(x="Country", y="mean(Payout)")
)

mean_po_overall = (
    alt.Chart(df)
    .mark_rule(color="blue", size=4, strokeDash=[3, 5])
    .encode(y="mean(Payout):Q")
)

annotation = (
    alt.Chart(df)
    .mark_text(align="left", baseline="middle", fontSize=12, dx=7, dy=-50, color="blue")
    .encode(text=alt.Text("mean(Payout):Q", format=",.0f"))
)

final_plot = (
    payouts + mean_po_by_country + mean_po_overall + annotation
).configure_axisX(labelAngle=0)

st.altair_chart(final_plot)
# %%
