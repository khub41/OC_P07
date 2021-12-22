import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

logged_model = 'runs:/a4a40c98e82c4913a6e66558e4665c94/best_model'
# shap.initjs()
data_test = pd.read_csv("data/data_test_scaled.csv", index_col=[0])

data_test = data_test.sample(100, random_state=41)


labels_test = data_test['TARGET'].copy()
data_test.drop(columns=['TARGET'], inplace=True)


best_model_mlflow = mlflow.sklearn.load_model(logged_model)

# explainer_shap = shap.TreeExplainer(best_model_mlflow)
# shap_values = explainer_shap.shap_values(data_test.iloc[[0]])
#
# # shap.summary_plot(shap_values, data_test, plot_type="bar")
#
# shap.force_plot(explainer_shap.expected_value[1], shap_values[1][0,:], data_test.iloc[0,:], matplotlib=True, show=True)

# TODO faire une liste des features interpretables

def get_explanation(id_client, model, strategy='normal'):

    if strategy.lower() == 'normal':
        threshold = 0.5
    elif strategy.lower() == 'aggressive':
        threshold = 0.8
    elif strategy.lower() == 'prudent':
        threshold = 0.2
    else:
        threshold = 0.5

    explainer_shap = shap.TreeExplainer(model)
    data_client = data_test.loc[[id_client]]

    shap_values_client = explainer_shap.shap_values(data_client)
    features = data_test.columns.values
    probas = model.predict_proba(data_client)

    if probas[0][1] > threshold:
        decision = 1
        litteral_decision = "rejects"
    else:
        decision = 0
        litteral_decision = "accepts"

    print(f"with a {strategy} strategy the model {litteral_decision} the client")

    explanation_client = pd.DataFrame({'shap_value': shap_values_client[decision][0, :],
                                       'feature_name': features})
    explanation_client.sort_values('shap_value',
                                   ascending=False,
                                   inplace=True)
    return explanation_client

client_decision_explained = get_explanation(200999, best_model_mlflow, strategy='normal')









# X = data_test[features_names].rename(columns=dictionnaire_features)
#
# shap_values_single = shap_kernel_explainer.shap_values(X.loc[id_mut_force_plot,:])
# plt.style.use('default') # Pour avoir le format standard matplolib (sinon une grille apparait)
# shap_values_single = shap_kernel_explainer.shap_values(X.loc[id_mut_force_plot,:])
# shap.force_plot(shap_kernel_explainer.expected_value,
#                 shap_values_single,
#                 X.loc[id_mut_force_plot,:].astype(int), # Pour avoir des valeurs arondies
#                 show=False,
#                 matplotlib=True).savefig('plots_storage/idmut_{}.png'.format(str(id_mut_force_plot)),
#                                          bbox_inches='tight')



# Global importance

df_importance = pd.Series(best_model_mlflow.feature_importances_, index=best_model_mlflow.feature_name_)
df_importance.sort_values(ascending=False, inplace=True)
df_importance_top = df_importance.head(15)
df_importance_top.sort_values(ascending=True, inplace=True)
plt.barh(df_importance_top.index,
         df_importance_top.values,
         color=plt.get_cmap("viridis").colors[0::round(256/15) + 1])
