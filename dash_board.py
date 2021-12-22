import json
import pandas as pd
import requests
import s3fs
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

fs = s3fs.S3FileSystem(anon=False)

st.set_page_config(layout="wide")


@st.cache
def load_scaled_data():
    with fs.open('homecreditdata/data_test_scaled.csv') as file:
        return pd.read_csv(file, index_col=[0], nrows=1000).drop(columns=["TARGET"])
    # return pd.read_csv("data/sample_test_scaled.csv", index_col=[0]).drop(columns=["TARGET"])


@st.cache
def load_raw_data():
    with fs.open('homecreditdata/data_raw_test.csv') as file:
        return pd.read_csv(file, index_col=[0]).drop(columns=["TARGET", "index"])
    # return pd.read_csv("data/data_full.csv", index_col=[0]).set_index('SK_ID_CURR')


@st.cache
def load_columns_descriptions():
    with fs.open('homecreditdata/HomeCredit_columns_description.csv') as file:
        desc = pd.read_csv(file, index_col=[0], encoding_errors='ignore')
        desc['table_pretty'] = desc.Table.replace({'application_{train|test}.csv': "application",
                                                   'bureau.csv': "bureau",
                                                   'bureau_balance.csv': "bureau balance",
                                                   'POS_CASH_balance.csv': "POS cash balance",
                                                   'credit_card_balance.csv': "credit card balance",
                                                   'previous_application.csv': "previous application",
                                                   'installments_payments.csv': "installments payments"})
        desc = desc.set_index('Row')
        return desc


# Importing Data from AWS s3
data_scale = load_scaled_data()
data_raw = load_raw_data()
column_descriptions = load_columns_descriptions()
# The user choses a client in the data base
id_client = st.sidebar.selectbox(
    "ID du client",
    data_scale.index
)

# Our API computes the risk. it posts the data about our client
url = "https://homecredit-oc-p7.herokuapp.com/predict"
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
data_client = [data_scale.loc[id_client].values.tolist()]
j_data = json.dumps(data_client)
r = requests.post(url, data=j_data, headers=headers)
risk = float(r.text.split('"')[1])

# The user chooses the strategy (it sets the threshold accordingly)
strategy = st.sidebar.selectbox(
    "Strategie",
    ["Prudente", "Normale", "Aggressive"]
)

if strategy.lower() == 'normale':
    threshold = 0.6

elif strategy.lower() == 'aggressive':
    threshold = 0.75

elif strategy.lower() == 'prudente':
    threshold = 0.4

else:
    threshold = 0.6

if risk > threshold:
    decision = 1
    litteral_decision = "rejette"
else:
    decision = 0
    litteral_decision = "accepte"

# The user chooses how many features he wants in the explanation graph
nb_features_explain = st.sidebar.slider(
    label='Nombre de critères pour expliquer la décision',
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# Title on the main page
st.title("Outil d'aide à la décision d'octroiement des prêts bancaires")

# Decision is displayed based on the risk and the strategy
st.header(
    f"Avec une stratégie {strategy.lower()}, le modèle {litteral_decision.upper()} le client"
)

# Init of two columns, one for a gauge showing the risk, the other for the explanation bar graph
col_score, col_explanation = st.columns(2)

with col_score:
    st.header(f"Risque calculé : {round(risk * 100, 2)} %")

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        mode="gauge",
        value=risk,
        title={'text': "Risque"},
        gauge={'axis': {'range': [None, 1]},
               'steps': [
                   {'range': [0, 0.4], 'color': "green"},
                   {'range': [0.4, 0.6], 'color': "orange"},
                   {'range': [0.6, 0.75], "color": "red"},
                   {'range': [0.75, 1], "color": "black"}, ],
               'bar': {'color': "blue"}
               }))

    st.plotly_chart(fig, use_container_width=True)

with col_explanation:
    st.header("Explication")
    st.caption("Les variables vertes font baisser le risque, les rouges l'augmentent!")

    # We need both scaled and raw data
    data_scale_client = data_scale.loc[[id_client]]
    data_raw_client = data_raw.loc[[id_client]]
    features = data_scale_client.columns.values

    # Sending the API the scaled data and getting a dict of the shap values
    url = "https://homecredit-oc-p7.herokuapp.com/explain"
    data_client = data_scale.loc[id_client].to_dict()
    response_api = requests.post(url, json=data_client)

    # We'll use a dataframe for convenience, sorting etc
    explanation_client = pd.DataFrame({'shap_value': response_api.json().values(),
                                       'feature_name': response_api.json().keys()})

    # We drop lines that are not "explainable" (extern score) but very efficient!
    explanation_client = explanation_client[~explanation_client.feature_name.isin(['EXT_SOURCE_1',
                                                                                   'EXT_SOURCE_2',
                                                                                   'EXT_SOURCE_3'])]
    # Getting most important lines using absolute values
    explanation_client['shap_value_abs'] = explanation_client.shap_value.map(abs)
    # Tagging positive and negative values and setting a color for plotting
    explanation_client['color'] = explanation_client.shap_value > 0
    explanation_client.color.replace(True, 'red', inplace=True)
    explanation_client.color.replace(False, 'green', inplace=True)
    # Sorting by abs value
    explanation_client.sort_values('shap_value_abs', ascending=False, inplace=True)
    # Getting only the number asked by user
    explanation_client = explanation_client.head(nb_features_explain)
    # Changing the order because plotly plots from bottom to top
    explanation_client.sort_values('shap_value_abs', ascending=True, inplace=True)
    # Getting raw data and writing it on the labels
    explanation_client['raw_data'] = data_raw_client[explanation_client.feature_name].iloc[0].values
    explanation_client['bar_labels'] = explanation_client.feature_name + '\n=' \
                                       + explanation_client.raw_data.round(2).astype(str)
    # Setup figure
    fig = go.Figure(go.Bar(x=explanation_client['shap_value'],
                           y=explanation_client['bar_labels'],
                           orientation='h',
                           marker={'color': explanation_client['color']},
                           ),
                    )
    fig.update_layout(xaxis_title="Influence sur le niveau de risque",
                      )

    st.plotly_chart(fig,
                    use_container_width=True)

# Aditional information about client features
st.header('En savoir plus sur mon client')
# About a feature
var_comparaison = st.selectbox(
    "Variable à explorer",
    data_raw.columns
)
# Getting the value and showing the value
var_comparaison_value = data_raw.loc[id_client][var_comparaison]
st.subheader(f"{var_comparaison}={var_comparaison_value}")

description_var = column_descriptions.loc[var_comparaison]
if type(description_var) == pd.core.frame.DataFrame:
    description_var = description_var.iloc[0]

st.caption(f"{description_var.Description} from {description_var.table_pretty}")
# Init to AMT_CREDIT
if var_comparaison == 'index':
    var_comparaison = "AMT_CREDIT"

use_mask = st.checkbox("Utiliser un filtre", value=False)
if use_mask:
    col_filter, col_logic, col_filter_value = st.columns(3)

    with col_filter:
        var_filter = st.selectbox('Variable à filtrer',
                                  data_raw.columns)
    with col_logic:
        logic_operator = st.selectbox('Opérateur logique',
                                      ['>', '>=', "=", "<", '<='],
                                      index=0)
    with col_filter_value:
        col_describe = data_raw[var_filter].describe()

        filter_value = st.number_input("Valeur",
                                       min_value=col_describe.loc['min'],
                                       max_value=col_describe.loc['max'],
                                       value=col_describe.loc['50%'])

# Setup histogram
if use_mask:
    if logic_operator == '>':
        masked_data = data_raw[data_raw[var_filter] > filter_value]
        fig_comparaison = px.histogram(masked_data[var_comparaison])
    elif logic_operator == ">=":
        masked_data = data_raw[data_raw[var_filter] >= filter_value]
        fig_comparaison = px.histogram(masked_data[var_comparaison])
    elif logic_operator == "=":
        masked_data = data_raw[data_raw[var_filter] == filter_value]
        fig_comparaison = px.histogram(masked_data[var_comparaison])
    elif logic_operator == "<":
        masked_data = data_raw[data_raw[var_filter] < filter_value]
        fig_comparaison = px.histogram(masked_data[var_comparaison])
    elif logic_operator == "<=":
        masked_data = data_raw[data_raw[var_filter] <= filter_value]
        fig_comparaison = px.histogram(masked_data[var_comparaison])
else:
    fig_comparaison = px.histogram(data_raw[var_comparaison])

# PLotting vertical line for position of client's feature compared to others
if not np.isnan(var_comparaison_value):
    fig_comparaison.add_vline(var_comparaison_value,
                              annotation_text=f'Client {id_client} \n {var_comparaison}={var_comparaison_value}',
                              annotation_position="top right",
                              line_dash="dot",
                              line_color='green',
                              line_width=3)
fig_comparaison.update_layout(xaxis_title=var_comparaison,
                              title='Repartition de la variable parmis les clients')

st.plotly_chart(fig_comparaison,
                use_container_width=True)

# col_var_x, col_var_y = st.columns(2)
#
# with col_var_x:
#     var_x = st.selectbox(
#         "Variable à tracer en X",
#         data_raw.columns
#     )
#
# with col_var_y:
#     var_y = st.selectbox(
#         "Variable à tracer en Y",
#         data_raw.columns
#     )
#
#
# st.plotly_chart(
#     go.Figure(go.Scatter(
#                          x=data_raw[var_x],
#                          y=data_raw[var_y]))
# )
