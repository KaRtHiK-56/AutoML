#streamlit 
import streamlit as sl 
# data manipulation library pandas 
import pandas as pd
# AutoMl Libraries
#1. Pycaret
#2. FLAML
from pycaret.classification import *
from pycaret.regression import *
from pycaret.regression import setup, compare_models, pull, save_model
from pycaret.classification import setup, compare_models, pull, save_model

#Auto EDA Libraries
#1. Pandas profiling
#2. SweetViz
#3. AutoViz
#4. Dtale
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
import dtale
import os

sl.markdown("""
<style>

.css-1w3peyr.egzxvld1
{
    visibility : hidden;
}


</style>
""" , unsafe_allow_html = True)


with sl.sidebar:
    sl.image("AUTOMLLOGO.png")
    sl.title("Welcome to the AutoML Webapp!!")
    menu=sl.radio("Menu" ,["Home" , "Upload" , "EDA" , "ML Model",  "Download" , "About Me"])

if menu=="Home":

    sl.title("**Auto Machine Learning Web Application**" )
    sl.header("AutoML Web App")
    sl.text("This is an Auto Machine learning web application which helps in making\nExploratary data analysis\
 and auto ML models in flexible and in easy manner\nto save your time.")
    sl.markdown("---")
    sl.subheader("Steps to Follow")
    listitems="""1. Click on the Upload option and upload the Dataset that you have.\n
2. After uploading the dataset you can see the dataset displayed in the current tab itself.\n
3. Now click on the EDA option to see the Exploratory Data Analysis of the data that you have uploaded.\n    
4. After the data analysis part the machine learning model has to be trained under the ML Model option.\n      
5. Run and see the results of the Auto ML Model."""
    sl.write(listitems)

if os.path.exists("source.csv"):
    df=pd.read_csv("source.csv",index_col=None)


if menu == "Upload":
    sl.title("Welcome to the Upload Section")
    sl.header("Upload your data for Analysis!")
    data = sl.file_uploader("Upload your dataset here.")
    if data:
       df = pd.read_csv(data,index_col=None,encoding="ISO-8859-1",error_bad_lines=False)
       df.to_csv("source.csv" ,index=None)
       sl.dataframe(df)

if menu == "EDA":
    sl.title("Welcome to the Data Analysis Section!")
    sl.header("This section will give your profile report.")
    edaselect = sl.selectbox("Select any of the Data analysis model for your Data from the given list."
        ,options=["Pandas-Profiling" , "SweetViz" , "DTale"])

    if edaselect == "Pandas-Profiling":
        if sl.button('Run EDA'):    
            pandas = df.profile_report()
            st_profile_report(pandas)
            

    if edaselect == "SweetViz":
        if sl.button('Run EDA'):
            sweetviz = sv.analyze(df)
            sweetviz.show_html('sweet_report.html')
            
    if edaselect == "DTale":
        if sl.button('Run EDA'):
            d = dtale.show(df)
            d.open_browser()

if menu == "ML Model":
    sl.title("Welcome to the Machine Learning Modelling Section!")
    sl.header("This section will predict on your data.")
    target = sl.selectbox('Choose the target variable that you want to predict from your data', df.columns)
    types = sl.selectbox("Select weather you want to perform Regression or Classification."
        ,options=["Regression" , "Classification"])


    if types == "Regression":
        if sl.button("Run Model"):
            setup(df, target = target)
            setup_df = pull()
            sl.dataframe(setup_df)
            best_model = compare_models(fold=5)
            compare_df = pull()
            sl.dataframe(compare_df)
            save_model(best_model,'best_model')

    if types == "Classification":
        if sl.button("Run Model"):
            setup(df, target = target)
            setup_df = pull()
            sl.dataframe(setup_df)
            best_model = compare_models(fold=5)
            compare_df = pull()
            sl.dataframe(compare_df)
            save_model(best_model,'best_model')


if menu == "Download":
    with open('best_model.pkl', 'rb') as f: 
        sl.download_button('Download Model', f , file_name="automl_model.pkl")
        sl.code("""After Downloading the model, please follow these instructions in your Coding Environment.\n
        from pycaret.classfication import load_model or\n
        from pycaret.regression import load_model
        model = load_model("automl_model")\nto see the model pipepline.""")  


if menu == "About Me":
    sl.title("@ ABOUT ME!!")
    sl.text("""Self-driven and Aspiring Engineer, Currently involved myself in the stream of 
Artificial intelligence.Passionate and Thriving Engineer Who Can Apply Techniques for data driven 
decision making and Knowledge to Develop and Solve Real-World Industry Problem by Adding Value to it.\n
My path for Artificial Intelligence started in 2020 where I was attracted to the way it functions. 
Since then my keen and interest in AI has never been down and wanted to contribute and provide solutions 
in this domain. For sure in the near future, I will be on my way contributing to this field and currently
working on becoming an Artificial Intelligence Engineer.Seeking to use my skills and knowledge to make
a positive impact as a data scientist.""")
    sl.markdown("[Portfolio](https://karthikpersonalportfolio56.000webhostapp.com/index.html)")
    sl.markdown("[Git-Hub](https://github.com/KaRtHiK-56)")
    sl.markdown("[Blog](https://sites.google.com/view/karthikaiblogs/home)")
    sl.markdown("[Personal Instagram](https://www.instagram.com/prince_6_karthik/)")
    sl.markdown("[Instagram](https://www.instagram.com/_.pythonista._/)")
    sl.markdown("[Medium](https://karthikvegeta.medium.com/)")
    sl.text("Gmail : karthiksurya611@gmail.com")
    sl.text("⚡ Fun fact: I got inspired by Tony Stark ✌️ and Motivated by 💪 Prince Vegeta ♕")