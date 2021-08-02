# Importation des packages
import numpy as np
import streamlit as st
import pandas as pd
from pickle import load

def main():
    st.title("Prédiction des départs de clients du service carte de credit")
    menu = ["Accueil", "Predictions", "Dataset"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice=="Predictions":
        st.subheader("Formulaire de saisie")

        with st.form(key='form1'):

            col1, col2 = st.beta_columns(2)

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
            def mi():
                return Total_Relationship_Count

            submitbouton = st.form_submit_button(label="predire")

        # Chargement du modèle
        model = load(open('model.pkl', 'rb'))

        if submitbouton:

            with st.beta_expander("Resultats"):
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
                   dd = {'Client existant':y_pred,'pourcentage de réussite':y_pred_proba[0,0]*100}
                elif (y_pred==1):
                   dd = {'Compte fermé ':y_pred,'pourcentage de réussite':y_pred_proba[0,1]*100}

                st.dataframe(dd)





          #  details = pd.DataFrame(dictionnaire, index=[0])


           # st.subheader("Valeurs prises : ")
            #st.write(mi())

    elif choice=="Dataset":

        st.subheader("Informations sur le jeu de données ")

        nom_dataset = st.sidebar.selectbox("Selectionner votre jeu de données", ["Credit card dataset"])
        #nom_classifier = st.sidebar.selectbox("Classifier", ["KNeigborsClassifier", "Regression Logistique", "Random Forest"])

        def get_dataset(name_dataset):
            if name_dataset=="Credit card dataset":
                dt = pd.read_csv("Dataset.csv", sep=";",  na_values="Unknown")

            return dt

        dt_credit = get_dataset(nom_dataset)
        st.write("Noms du jeu de données : ", nom_dataset)
        st.write("Les dimensions initiales du jeu de données : ", dt_credit.shape)
        st.write("Nombre de classes à prédire : ", len(np.unique(dt_credit['Attrition_Flag'])))
        st.write("Les différentes classes : ", np.unique(dt_credit['Attrition_Flag']))
        #model_random=load(open('model.pkl', 'rb'))

        #def afficher_score(clf_classifier):
            #if clf_classifier=="Random Forest":
                #st.write("Score du modèle de Random Forest", model_random.score())


       # afficher_score(nom_classifier)



    else:
        #st.subheader("")

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

        st.subheader("Les différentes caractéristiques retenues : ")
        st.write(df.T)


if __name__=='__main__':
    main()