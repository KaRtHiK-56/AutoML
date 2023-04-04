import streamlit as sl 
import pandas as pd
from sklearn.impute import SimpleImputer
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
from mlbox import mlb
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os


with sl.sidebar:
    sl.image("AUTOMLLOGO.PNG")
    sl.title("Welcome to the AutoML Webapp!!")
    menu=sl.radio("Menu" ,["Home","Upload","EDA","ML Model","About Me"])

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
    report = df.profile_report()
    st_profile_report(report)

if menu == "ML Model":
    sl.title("Welcome to the Machine Learning Modelling Section!")
    sl.header("This section will predict on your data.")
    target = sl.selectbox('Choose the target variable that you want to predict from your data', df.columns)
    if sl.button('Auto ML'):
        model = mlb.preprocessing.Drift_thresholder().fit_transform(df)
        evaluate = mlb.optimisation.Optimiser().evaluate(None, model)
        space = {

        'ne__numerical_strategy' : {"space" : [0, 'mean']},

        'ce__strategy' : {"space" : ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy' : {"space" : ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search" : "choice", "space" : [0.1, 0.2, 0.3]},

        'est__strategy' : {"space" : ["LightGBM"]},
        'est__max_depth' : {"search" : "choice", "space" : [5,6]},
        'est__subsample' : {"search" : "uniform", "space" : [0.6,0.9]}

        }

        best = mlb.optimise(space, evaluate, max_evals = 5)
        comparision = mlb.prediction.Predictor().fit_predict(best, eval)
        sl.dataframe(comparision)
        

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