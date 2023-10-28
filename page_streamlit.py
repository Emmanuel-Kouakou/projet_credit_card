# Importation des packages
import numpy as np
import streamlit as st
import pandas as pd
from pickle import load
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import sqlite3


#Utiles
import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

## Télécharger text
def telecharge_text(text):
    b64 = base64.b64encode(text.encode()).decode()
    nom_new_fichier = "nouveau_fichier_text_{}_.txt".format(timestr)
    st.markdown("### Téléchager le fichier ###")
    href = f'<a href="data:file/txt;base64,{b64}" download="{nom_new_fichier}">Cliquer ici !! </a>'
    st.markdown(href, unsafe_allow_html=True)

 ## Télécharger CSV
def telecharge_csv(data):
    fichier_csv = data.to_csv()
    b64 = base64.b64encode(fichier_csv.encode()).decode()
    fichier_telecharge = "prediction_client_{}_.csv".format(timestr)
    st.markdown("### Téléchager le fichier en format csv ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{fichier_telecharge}"> Cliquer ici !! </a>'
    st.markdown(href, unsafe_allow_html=True)


# Chargement du modèle
model = load(open('model.pkl', 'rb'))

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

def view_table():
    c.execute("SELECT CLIENTNUM, Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio, Attrition_Flag, Attrition_Flag_Predict  FROM data_credit_card LIMIT 100")
    data = c.fetchall()
    return data


def select_aleatoire(nbre):
    c.execute('SELECT Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio, Attrition_Flag FROM data_credit_card ORDER BY RANDOM() LIMIT "{}"'.format(nbre))
    data = c.fetchall()

    return data


## FONCTION POUR MANIPULER LE DATAFRAME
def edit_dataframe(data):
    ## Pour selectionner des lignes et souvegarder les lignes sélectionnées
    update_mode_value = GridUpdateMode.MODEL_CHANGED

    gb = GridOptionsBuilder.from_dataframe(data)

    # customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum')

    gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True)

    ######
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    #####

    grid_response = AgGrid(data,
                           gridOptions=gridOptions,
                           width='100%',
                           update_mode=update_mode_value,
                           enable_enterprise_modules='Enable Enterprise Modules'
                           )


    df = grid_response['data']

    ## Les valeurs sélectionnées de mon Dataframe
    selected = grid_response['selected_rows']
    selected_df = pd.DataFrame(selected)

    return df, selected,selected_df


## Lire et predire tous le dataset

def read_predict(df):

    print("OK Normalement")
    data = df.copy()

    colonne_selectionne = ["CLIENTNUM","Customer_Age",
                           "Total_Relationship_Count",
                           "Credit_Limit",
                           "Total_Revolving_Bal",
                           "Total_Amt_Chng_Q4_Q1",
                           "Total_Trans_Amt",
                           "Total_Trans_Ct",
                           "Total_Ct_Chng_Q4_Q1",
                           "Avg_Utilization_Ratio","Attrition_Flag"]

    liste_bool = []
    for col in colonne_selectionne:
        if col in data.columns:
            liste_bool.append(True)
        else:
            liste_bool.append(False)

    if all(liste_bool) == False:
        st.warning("Opération impossible, le fichier que vous avez chargé ne contient pas des colonnes adéquats pour la prédiction.")
    else:
        st.success("Opération réussie, Chargement du fichier...")

        data = data[colonne_selectionne]

        dt, selected, selected_df = edit_dataframe(data)

        #st.write(selected)
        bouton = st.button("predire")

        if bouton:

            #Z = selected_df.copy()
            X = selected_df.drop(["CLIENTNUM","Attrition_Flag"], axis=1)
            #
            # y = selected_df["Attrition_Flag"]
            #
            ypred = model.predict(X)
            #
            liste_pred = []
            for pred in ypred:
                if pred == 0:
                    msg = "Client existant"
                elif pred == 1:
                    msg = "Client fermé"

                liste_pred.append(msg)

            st.subheader("Predictions des individus sélectionnés :")

            de = pd.concat([selected_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
            AgGrid(de)


            telecharge_csv(de)



def main():
    #html_temp = """<div style = "background-color:blue;padding:10px"><h2 style = "color:white;text-align:center;"> CHURN PREDICTION </h2></div>"""
    #st.markdown(html_temp, unsafe_allow_html=True)

    st.title("Prédiction des départs de clients du service carte de credit")

    st.sidebar.write("Menu")
    choice = st.sidebar.radio("",("Accueil", "Prediction unique", "Predictions à partir du Dataset","Charger un fichier"))

    if choice=="Prediction unique":
        st.subheader("Formulaire de saisie")

        with st.form(key='form1'):
            col1, col2 = st.columns(2)

            with col1:
                Customer_Age = st.number_input("Age du client (Customer Age)", min_value=26, max_value=73, step=1)
                Total_Trans_ct = st.number_input("Nombre total de transaction (Total Trans Ct)", min_value=10, max_value=139, step=1)
                Credit_Limit = st.slider(label="Limite de crédit sur la carte de crédit (Credit Limit)", min_value=1438.3,max_value=34516.0, value=20510.0)
                Total_Amt_Chng_Q4_Q1 = st.slider(label="Changement du montant de la transaction T4-T1 (Total Amt Chng Q4_Q1)", min_value=0.0, max_value=3.397, value=1.5)

            with col2:
                Total_Revolving_Bal = st.number_input("Solde Renouvelable total sur la carte de credit (Total Revolving Bal)", min_value=0, max_value=2517, step=1)
                Total_Trans_Amt = st.number_input("Montant total de la transaction (Total Trans Amt)", min_value=510, max_value=18484,step=1)
                Total_Ct_Chng_Q4_Q1 = st.slider(label="Changement du nombre de transaction T4-T1 (Total Ct Chng Q4_Q1)", min_value=0.0,max_value=3.714, value=1.0)
                Avg_Utilzation_Ratio = st.slider(label="Taux d'utilisation moyen de la carte (Avg Utilization Ratio)", min_value=0.0,max_value=0.99, value=0.5)

            Total_Relationship_Count = st.selectbox("Nombre de produits détenus par le client (Total Relationship Count)", [1, 2, 3, 4, 5, 6])


            submitbouton = st.form_submit_button(label="predire")



        if submitbouton:

            with st.expander("Resultats"):
                valeurs = np.array([Customer_Age,
                                Total_Relationship_Count,
                                Credit_Limit,
                                Total_Revolving_Bal,
                                Total_Amt_Chng_Q4_Q1,
                                Total_Trans_Amt,
                                Total_Trans_ct,
                                Total_Ct_Chng_Q4_Q1,
                                Avg_Utilzation_Ratio]).reshape(1,9)

                y_pred = model.predict(valeurs)
                y_pred_proba = model.predict_proba(valeurs)
                if y_pred == 0 :
                   dd = {'Client existant':y_pred,'pourcentage':y_pred_proba[0,0]*100}
                elif (y_pred==1):
                   dd = {'Possibilité de départ':y_pred,'pourcentage':y_pred_proba[0,1]*100}

                #st.success('')

                st.dataframe(dd)



   ### PREDICTIONS A PARTIR DU JEU DE DONNEES

    elif choice == "Predictions à partir du Dataset":

        st.subheader("Prédiction à partir du jeu de données ")

        st.subheader("Rechercher")

        # Rechercher un nombre d'individus de la Bd et fais leur predictions IMPORTANT

        nombre=st.number_input("Entrer un nombre de prédictions à effectuer", min_value=1, step=1)
        rechercher = st.button('charger')

        if rechercher:
            resul_rech_al = select_aleatoire(nombre)
            result_al_df = pd.DataFrame(resul_rech_al,columns=["Customer_Age", "Total_Relationship_Count", "Credit_limit","Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Valeur Observée"] )
            #st.dataframe(result_al_df)

            liste_pred = []
            for i in range(result_al_df.shape[0]):
               X = result_al_df.iloc[i,:-1]


               X=np.array(X).reshape(1,-1)

               ypred = model.predict(X)
               #st.write(ypred)

               if ypred==0:
                   msg="Client existant"
               elif ypred==1:
                   msg="Client fermé"

               liste_pred.append(msg)
            #st.write(liste_pred)
            #st.dataframe(pd.DataFrame(liste_pred, columns=["Predictions"]))

            de=pd.concat([result_al_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
            AgGrid(de)

        # with col2:
        #     res = st.button('predire')
        #     ls=[]
        #     for i in data_t:
        #         if st.checkbox(f"{i[0]}"):
        #             ls.append(i[0])
        #
        #     if res:
        #         for value in ls:
        #             c.execute('SELECT Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio FROM data_credit_card WHERE CLIENTNUM="{}"'.format(value))
        #             data = c.fetchall()
        #             ypred=model.predict(data)
        #
        #             if ypred == 0:
        #                 msg = 'Client_existant'
        #             elif ypred == 1:
        #                 msg = 'Client_ferme'
        #             c.execute('UPDATE data_credit_card SET Attrition_Flag_Predict="{}" WHERE CLIENTNUM={}'.format(msg, value))
        #             conn.commit()
        #         ls.clear()
        #
        #         st.write(ls)


   ### CHARGER FICHIER

    elif choice=="Charger un fichier":

        st.subheader("Charger fichier csv, excell, txt")
        #st.subheader("Charger fichier csv, excell, txt")
        csv_excell_file = st.file_uploader("Charger ici", type=["csv","xls", "txt"])
        if csv_excell_file is not None:


            details_fichier = {
                "Nom du fichier": csv_excell_file.name,
                "Type du fichier": csv_excell_file.type,
                "Taille du fichier": csv_excell_file.size
            }
            st.write(details_fichier)
            if csv_excell_file.type == "text/csv": 
            # if csv_excell_file.type == "application/vnd.ms-excel":
                df = pd.read_csv(csv_excell_file, na_values=['Unknown'])
                #st.write(df)
                read_predict(df)


            elif csv_excell_file.type == "text/plain":
                df = pd.read_table(csv_excell_file)
                read_predict(df)




   ### ACCUEIL

    else:

        st.image(image='images/predict_card2.jpg')

        st.subheader("CONTEXTE")

        st.write("Un responsable d’une banque souhaite réduire le nombre de clients qui quittent leurs services  de carte de crédit. Il aimerait pouvoir anticiper le départ des clients afin de leur fournir de meilleurs services et ainsi les retenir.")

        st.subheader("OBJECTIFS")

        st.write("Mettre en place un modèle de Machine Learning capable de prédire les départs des clients.")

        st.subheader("DESCRIPTION DU JEU DE DONNEES :")

        nom_dataset = st.sidebar.radio("Selectionner votre jeu de données", ["Credit card dataset"])

        # nom_classifier = st.sidebar.selectbox("Classifier", ["KNeigborsClassifier", "Regression Logistique", "Random Forest"])

        def get_dataset(name_dataset):
            if name_dataset == "Credit card dataset":
                dt = pd.read_csv("Dataset.csv", sep=";", na_values="Unknown")

            return dt

        dt_credit = get_dataset(nom_dataset)
        st.write("Noms du jeu de données : ", nom_dataset)
        st.write("Les dimensions initiales du jeu de données : ", dt_credit.shape)
        st.write("Nombre de classes à prédire : ", len(np.unique(dt_credit['Attrition_Flag'])))
        st.write("Les différentes classes : ", np.unique(dt_credit['Attrition_Flag']))




        def afficher_caracteristique():
            dictionnaire = {
                'Customer_Age': 'Age du client',
                'Total_Relationship_Count': 'Nombre total de produits détenus par le client',
                'Credit_Limit': 'Limite de crédit sur la carte de crédit',
                'Total_Revolving_Bal': 'Solde renouvelable total sur la carte de crédit',
                'Total_Amt_Chng_Q4_Q1': 'Changement de montant de transaction T4 par rapport à T1',
                'Total_Trans_Amt': 'Montant total de transactions',
                'Total_Trans_ct': 'Nombre total de transactions',
                'Total_Ct_Chng_Q4_Q1': 'Changement du nombre de transactions',
                'Avg_Utilzation_Ratio': "Taux d'utilisation moyen de la carte"
            }

            infos = pd.DataFrame(dictionnaire, index=[0])

            return infos

        df=afficher_caracteristique()

        st.subheader("Les différentes caractéristiques retenues pour la modélisation : ")
        st.write(df.T)


if __name__=='__main__':
    main()