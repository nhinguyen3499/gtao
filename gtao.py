### Monitor time execution of the model
import time
start_time = time.time()

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import metrics as mt
import matplotlib.pyplot as plt

### The first two lines of the code load an image and display it using the st.image function
image = Image.open('gtao.png')
st.image(image)

### The st.sidebar function creates a sidebar and adds a header and a markdown line. The app_mode variable is created to hold the selected option from the dropdown menu.
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])

if app_mode == 'Introduction': 
### The select_dataset variable is created to hold the selected dataset from the dropdown menu. The pd.read_csv function is used to read the data from a CSV file.
  select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Recorded_stats"])
  df = pd.read_csv("recorded_stats.csv")
  st.title("Grand Theft Auto Online Dashboard - 01 Introduction Page 🔎")

  ### The st.markdown function is used to display some text and headings on the web application.
  st.markdown("### 00 - Show  Dataset")

  ### The st.number_input function creates a widget that allows the user to input a number. The st.radio function creates a radio button widget that allows the user to select either "Head" or "Tail".
  num = st.number_input('No. of Rows', 5, 10)
  head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
  if head == 'Head':
  ### the st.dataframe function displays the data frame.
    st.dataframe(df.head(num))
  else:
    st.dataframe(df.tail(num))

  st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
  ### The st.text and st.write functions display the shape of the data frame and some information about the variables in the data set.
  st.text('(Rows,Columns)')
  st.write(df.shape)

  st.markdown("##### Variables ➡️")
  st.markdown(" **account_id**: Unique account ID tied to the platform")
  st.markdown(" **platform_id**: Platform indicator (PC,PS4,XBOX)")
  st.markdown(" **occur_date**: Date in field in the format YYYY-MM-DD (day of occurence")
  st.markdown(" **activity_type**: Broad activity category (Heist, Races, Biker Missions,...)")
  st.markdown(" **time_spent**: Time spent in the activity (hours)")
  st.markdown(" **kills**: Number of kills during activity")
  st.markdown(" **deaths**: Number of deaths during activity")
  st.markdown(" **suicides**: Number of suicides during activity")
  st.markdown(" **money_earned**: GTA$ earned in activity ")
  st.markdown(" **rp_earned**: RP (experience) earned in activity")
  st.markdown(" **success**: Indicator of a sucessful activity conclusion")
  st.markdown(" **item**: Item name as seen in game")
  st.markdown(" **item_type**: Item type descriptor (vehicle, property, weapon)")
  st.markdown(" **item_sub_type**: Secondary item descriptor (car, helicopter, garage,...)")
  st.markdown(" **money_spent**: GTA$ spent")

  ### The st.markdown and st.dataframe functions display the descriptive statistics of the data set.
  st.markdown("### 01 - Description")
  st.dataframe(df.describe())

  ### The st.markdown, st.write, and st.warning functions are used to display information about the missing values in the data set.
  st.markdown("### 02 - Missing Values")
  st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**. Original dataset from Rockstar North Limited - development studio of Grand Theft Auto Online game")
  dfnull = df.isnull().sum()/len(df)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of total missing values:",totalmiss)
  st.write(dfnull)
  if totalmiss <= 30:
    st.success("Looks good! as we have less then 30 percent of missing values.")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")
if app_mode == "Visualization":
  ### The st.title() function sets the title of the Streamlit application to "Midterm Project - Cardekho's Used Cars Visualization".
  st.title("Grand Theft Auto Online Dashboard - 02 Visualization Page 📊")

  ### The pd.read_csv() function loads a CSV file of Grand Theft Auto Online's activity played data into a Pandas DataFrame called "df".
  df = pd.read_csv('recorded_stats.csv')

  ### Create report on streamlit
  from dataprep.eda import create_report
  create_report(df)
  from pandas_profiling import ProfileReport
  profile = ProfileReport(df, title="Report")
  st.components.v1.html(profile.to_html(), width=1000, height=500, scrolling=True)

  ### Boxplot: relationship between rp_earned and success
  st.subheader('Boxplot of Experience Earned vs. Success and Non-Success 💡')
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.boxplot(x="success", y="rp_earned", data=df, hue="success", palette='rainbow', ax=ax)
  ax.set_xlabel('success', fontsize=12)
  ax.set_ylabel('rp_earned', fontsize=12)
  ax.set_ylim([0, 5000])  # set y-axis limits
  ax.grid(True)
  ax.set_title('Boxplot of Experience Earned vs. Success and Non-Success', fontsize=18)
  ax.legend(title='success', loc='upper left')
  st.pyplot(fig)
  plt.show()
  
if app_mode == "Prediction":
  ### The st.title() function sets the title of the Streamlit application to "Mid Term Template - 03 Prediction Page 🧪".
  st.title("Grand Theft Auto Online Dashboard - 03 Prediction Page 🔋")
  org_df = pd.read_csv("recorded_stats.csv")
  df = org_df.dropna()

  ### Create a tab for users to choose model
  model_type = st.sidebar.selectbox('🔎 Select Model',['Tree Classifier','K_Nearest_Neighbor'])

  ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.
  test_size = st.sidebar.number_input("Test Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.30)
  
  if model_type == 'Tree Classifier':
    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    list_variables = ['success']
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns

    ### View of dataset
    st.write("Dataset")
    st.write(df)

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    output_multi = st.multiselect("Select Explanatory Variables", list_var, default=['time_spent','kills','money_earned','rp_earned'])
  
    new_df2 = new_df[output_multi]
    x = new_df2
    y = df['success']

    ### The train_test_split() function splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    ### Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    ### The clf.fit() function fits the Decision Tree model to the training data.
    clf.fit(X_train,y_train)

    ###The clf.predict() function generates predictions for the testing data.
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    ### The st.columns() function creates 3 columns to display the feature columns and target column.
    col1,col2 = st.columns(2)
    col1.subheader("Variables used")
    col1.write(X_test.head(10))
    col2.subheader("Prediction vs Real data")

    ###Concat 2 columns, covert predictions type from array to df
    pred = pd.DataFrame(y_pred)
    y_test.reset_index(drop=True,inplace=True)
    pred.reset_index(drop=True,inplace=True)
    df_pred = pd.concat([pred,y_test],ignore_index=True,axis=1) 
    df_pred.columns = ['Prediction','Real data']
    col2.write(df_pred.head(10))

    ### The st.subheader() function creates a subheading for the results section.
    st.subheader('🎯 Results')
    st.write("1) The Accuracy score of model is:", np.round(mt.accuracy_score(y_test, y_pred),2))
    st.write("2) Confusion matrix")

    ###Confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8,6), dpi=50)
    display = ConfusionMatrixDisplay(conf_matrix)
    ax.set(title='Confusion Matrix for Success Prediction Model')
    display.plot(ax=ax)
    st.pyplot(fig)
    
  else:
    ### create a sidebar for n_neighbor number
    n_number = st.sidebar.number_input("N_neighbor", min_value= 0, step= 1, max_value= 40, value= 4)

    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    list_variables = ['activity_type']
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    ###Standardize the variable
    from sklearn.preprocessing import StandardScaler 
    df['platform_id'] = df['platform_id'].factorize()[0]
    df['activity_type'] = df['activity_type'].factorize()[0]
    df['item'] = df['item'].factorize()[0]
    df['item_type'] = df['item_type'].factorize()[0]
    df['item_sub_type'] = df['item_sub_type'].factorize()[0]

    ###Scaling
    df.drop(['occur_date','account_id'], axis=1, inplace=True)
    scaler = StandardScaler()
    scaler.fit(df.drop(['activity_type'],axis=1))
    scaled_features = scaler.transform(df.drop('activity_type',axis=1))
    cols = ['platform_id', 'time_spent', 'kills', 'deaths','suicides', 'money_earned', 'rp_earned', 'success', 'item', 'item_type','item_sub_type', 'money_spent']
    df_feat = pd.DataFrame(scaled_features,columns=cols)

    ### View of dataset after dropping and factorizing columns
    st.write("Dataset")
    st.write(df_feat.head())

    list_var = df_feat.columns

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    output_multi = st.multiselect("Select Explanatory Variables", list_var, default=['money_earned','rp_earned','kills','deaths','suicides','time_spent','success'])
    new_df2 = df_feat[output_multi]
    x = new_df2
    y = df['activity_type']

    ## The train_test_split() function splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_number)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    ### The st.columns() function creates 3 columns to display the feature columns and target column.
    col1,col2 = st.columns(2)
    col1.subheader("Variables used")
    col1.write(X_test.head(10))
    col2.subheader("Prediction vs Real data")

    ###Concat 2 columns
    pred = pd.DataFrame(y_pred)
    y_test.reset_index(drop=True,inplace=True)
    pred.reset_index(drop=True,inplace=True)
    df_pred = pd.concat([pred,y_test],ignore_index=True,axis=1) 
    df_pred.columns = ['Prediction','Real data']
    col2.write(df_pred.head(10))

    ### The st.subheader() function creates a subheading for the results section.
    st.subheader('🎯 Results')
    st.write("The Accuracy score of model is:", np.round(mt.accuracy_score(y_test, y_pred),2))

  ### Monitor time execution of the model 
  import time
  start_time = time.time()
  st.write("--- %s seconds ---" % (np.round(time.time() - start_time,2)))

  ### Monitor carbon emissions of the model 
  from codecarbon import OfflineEmissionsTracker
  tracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
  tracker.start()
  dtree = DecisionTreeClassifier()

  # Train Decision Tree Classifer
  dtree = dtree.fit(X_train,y_train)

  results = tracker.stop()
  st.write(' %.12f kWh' % results)
  ### Monitor time
  st.subheader('🎯 Time execution') 
  st.write("Time execution of the model is --- %s seconds ---" % (np.round(time.time() - start_time,2)))

  ### Monitor carbon emissions of the model 
  from codecarbon import OfflineEmissionsTracker
  tracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
  tracker.start()
  dtree = DecisionTreeClassifier()

  ###Train Decision Tree Classifer
  st.subheader('🎯 Electricity consumption') 
  dtree = dtree.fit(X_train,y_train)
  results = tracker.stop()
  st.write('The electricity consumption of the model is %.12f kWh' % results)
