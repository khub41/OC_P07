import mlflow
import pandas as pd
import s3fs
import streamlit as st
import shap

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
    with fs.open('homecreditdata/data_full.csv') as file:
        return pd.read_csv(file, index_col=[0]).set_index('SK_ID_CURR')
    # return pd.read_csv("data/data_full.csv", index_col=[0]).set_index('SK_ID_CURR')


model_path = 'best_model'
model = mlflow.sklearn.load_model(model_path)


# Create a text element and let the reader know the data is loading.


# data_load_state = st.text('Loading data...')
data_scale = load_scaled_data()
data_raw = load_raw_data()
# data_load_state.text('Loading data...done!')

id_client = st.sidebar.selectbox(
    "ID du client",
    data_scale.index
)

risk = model.predict_proba(data_scale.loc[[id_client]])[0][1]

strategy = st.sidebar.selectbox(
    "Strategie",
    ["Prudente", "Normale", "Aggressive"]
)

nb_features_explain = st.sidebar.slider(
    label='Nombre de critères pour expliquer la décision',
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.title("Outil d'aide à la décision d'octroiement des prêts bancaires")

if strategy.lower() == 'normale':
    threshold = 0.5
    # article = "a"
elif strategy.lower() == 'aggressive':
    threshold = 0.8
    # article = "an"
elif strategy.lower() == 'prudente':
    threshold = 0.2
    # article = "a"
else:
    threshold = 0.5
    # article = "a"

if risk > threshold:
    decision = 1
    litteral_decision = "rejette"
else:
    decision = 0
    litteral_decision = "accepte"


st.header(
    f"Avec une stratégie {strategy.lower()}, le modèle {litteral_decision.upper()} the client"
)


col_score, col_explanation = st.columns(2)


with col_score:

    st.header(f"Risque calculé {round(risk, 2)}")

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        mode="gauge",
        value=risk,
        title={'text': "Risk"},
        gauge={'axis': {'range': [None, 1]},
               'steps': [
                   {'range': [0, 0.2], 'color': "lightgreen"},
                   {'range': [0.2, 0.5], 'color': "orange"},
                   {'range': [0.5,0.8], "color": "red"},
                   {'range': [0.8,1], "color": "black"}] ,
               }))

    st.plotly_chart(fig, use_container_width=True)

with col_explanation:
    st.header("Explication")
    st.caption("Les variables bleues font baisser le risque, les rouges l'augmente!")

    explainer = shap.TreeExplainer(model)

    explainer_shap = shap.TreeExplainer(model)
    data_scale_client = data_scale.loc[[id_client]]
    data_raw_client = data_raw.loc[[id_client]]
    features = data_scale_client.columns.values
    shap_values_client = explainer_shap.shap_values(data_scale_client)

    explanation_client = pd.DataFrame({'shap_value': shap_values_client[1][0, :],
                                       'feature_name': features})

    explanation_client = explanation_client[~explanation_client.feature_name.isin(['EXT_SOURCE_1',
                                                                                   'EXT_SOURCE_2',
                                                                                   'EXT_SOURCE_3'])]

    # if decision == 1:
    #     # Reject the client
    #     ascending_bool = False
    # else:
    #     ascending_bool = True

    explanation_client['shap_value_abs'] = explanation_client.shap_value.map(abs)
    explanation_client['color'] = explanation_client.shap_value > 0
    explanation_client.color.replace(True, 'red', inplace=True)
    explanation_client.color.replace(False, 'blue', inplace=True)
    explanation_client.sort_values('shap_value_abs', ascending=False, inplace=True)


    explanation_client = explanation_client.head(nb_features_explain)
    explanation_client.sort_values('shap_value_abs', ascending=True, inplace=True)
    explanation_client['raw_data'] = data_raw_client[explanation_client.feature_name].iloc[0].values
    explanation_client['bar_labels'] = explanation_client.feature_name + '\n=' \
                                       + explanation_client.raw_data.round(2).astype(str)

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

st.header('En savoir plus sur mon client')

var_comparaison = st.selectbox(
        "Variable à explorer",
        data_raw.columns
    )
var_comparaison_value = data_raw.loc[id_client][var_comparaison]

st.subheader(f"{var_comparaison}={var_comparaison_value}")
if var_comparaison == 'index':
    var_comparaison = "AMT_CREDIT"
fig_comparaison = px.histogram(data_raw[var_comparaison])
fig_comparaison.add_vline(var_comparaison_value,
                          annotation_text=f'Client {id_client} \n {var_comparaison}={var_comparaison_value}',
                          annotation_position="top right",
                          line_dash="dot",
                          line_color='green',
                          line_width=3)
fig_comparaison.update_layout(xaxis_title=var_comparaison,
                              title='Repartition de la variable parmis les client')

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
