import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import squarify
import seaborn as sns

### Page setup ###
st.set_page_config(page_title="Analysis Dashboard", layout="wide")
st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>',unsafe_allow_html=True) #ปรับ top padding
st.title("Insight of RFM model")
# End Page setup #

### Sidebar ###
st.markdown("""
<style>
[data-testid=stSidebar] {
    background-color:rgb(41, 49, 68);
    background-size: cover;
    color: black;
}
</style>
""", unsafe_allow_html=True)
#background-image: url("https://aboffs.com/cdn/shop/products/2066-60-honolulublue_37713baa-8046-469c-bff5-f36653a376cb_1600x.png?v=1587091011");

with st.sidebar:
    st.header("Please take a questionnaire: ")

### Questionnaire ###
    type = st.radio("What is your business type?", ["Agricultural production", "Industrial production", "Service", "Retail", "Wholesale"],index=None,)

    focus = st.radio("What is your company's main focus?", 
                    ["Customers", "Affordable prices", "Product quality"],index=None,)

    target = st.radio("What is the target audience?",
                    ["Child", "Teenager", "Adult", "absolutely"],index=None,)

    channel = st.radio("Main sales channel", ["Store/Shop", "Online"],index=None,)

    expect = st.radio("What do you expect the most from customer behavior analysis?",
                    ["To increase sales", "To increase customer loyalty", "To create personal branding"],index=None,)
    # End Questionnaire #

    ### Upload file ###
#     st.markdown('''
# :violet-background[:red[Please changes name the columns in your file.]]<br>
# :orange[The ID of customer: customer_ID]<br>
# :green[The duration that has passed since the customer last made a purchase or utilized a service: recency]<br>
# :blue[The frequency of customers purchasing goods or using services: frequency]<br>
# :violet[The amount of money spent on products or customer service: Monetary]
# ''', unsafe_allow_html=True)

    @st.cache_data
    def load_data(file):
        data = pd.read_csv(file)
        return data

    uploaded_file = st.file_uploader("Choose a file")
    
    def CleansingData(uploaded_file):
        if uploaded_file.name == 'OnlineRetail.csv':
            df = load_data(uploaded_file)
            df = df[df['CustomerID'].notnull()]

            for col in ['Quantity', 'UnitPrice']:
                series = sorted(df[col])
                Q1, Q3 = np.quantile(series, [0.01, 0.99])
                IQR = Q3 - Q1
                lowerLimit = Q1 - (1.5 * IQR)
                upperLimit = Q3 + (1.5 * IQR)
                df[col] = np.where(df[col] < lowerLimit, lowerLimit, df[col])
                df[col] = np.where(df[col] > upperLimit, upperLimit, df[col])
            
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
            fixDate = np.max(df['InvoiceDate'])

            df = df.eval("TotalPrice = Quantity * UnitPrice")

            Data_clean = df.groupby(['CustomerID']).agg(
                {
                    'InvoiceDate': lambda date: (fixDate - date.max()).days,
                    'InvoiceNo': lambda num: num.nunique(),
                    'TotalPrice': lambda price: price.sum()
                }
            )
            Data_clean.columns = ['recency', 'frequency', 'monetary']
            return Data_clean
        else:
            df = load_data(uploaded_file)
            return df
        

    # ENd Upload file #

    submit = st.button("SUBMIT", type="primary")
# End Sidebar #


### Main Page ###
def RFMmodel(df):
    Data_clean = pd.concat([df['recency'], df['frequency'],df['monetary']], axis=1)
    RFM_data =  Data_clean
    RFM_data['RecencyScore'] = pd.qcut(RFM_data['recency'], 5, labels = [5, 4, 3, 2, 1])
    RFM_data['FrequencyScore'] = pd.qcut(RFM_data['frequency'].rank(method = 'first'),5, labels = [1, 2, 3, 4, 5])
    RFM_data['MonetaryScore'] = pd.qcut(RFM_data['monetary'], 5, labels = [1, 2, 3, 4, 5])
    RFM_data['FMScore'] = (RFM_data['FrequencyScore'].astype('float') + RFM_data['MonetaryScore'].astype('float'))/2
    RFM_data['FMScore'] = np.ceil(RFM_data['FMScore']).astype('int')
    RFM_data['RFMScore'] = RFM_data['RecencyScore'].astype('str') + RFM_data['FMScore'].astype('str') 




    # Convert RFM Score to segment label
    RFMLabel = {
        r'[5][4-5]': "Champion",
        r'52': "Recent User",
        r'51': "Price Sensitive",
        r'[4-5][2-3]': "Potential Loyalist",
        r'41': "Promising",
        r'[3-4][4-5]': "Loyal Customer",
        r'33': "Needs Attention",
        r'[3][1-2]': "About to Sleep",
        r'[1-2][5]': "Can't Lose Them",
        r'[1-2][3-4]': 'Hibernating',
        r'[1-2][1-2]': "Lost"
    }
    RFM_data['Segment'] = RFM_data['RFMScore'].replace(RFMLabel, regex = True)


    result = RFM_data.groupby(['Segment'])['Segment'].count()
    values = list(result)
    labels = result.index
    total_customers = len(RFM_data)
    labels_with_percentage = [f"{label} ({value / total_customers * 100:.1f}%)" for label, value in zip(labels, values)]

    # Plot
    colors = ['#070F2B', '#1B1A55', '#535C91', '#9290C3']
    plt.figure(figsize = (18, 11), facecolor='none')  
    squarify.plot(sizes=values, color=colors, label=labels_with_percentage ,text_kwargs={'color': 'white', 'fontsize': 12})
    plt.title('Customer segmentation',color='white',fontsize=16)
    plt.xlabel('Recency',color='white', fontsize=16)
    plt.ylabel('FMScore',color='white', fontsize=16)
    plt.tick_params(axis='x', colors='white', labelsize=14)  
    plt.tick_params(axis='y', colors='white', labelsize=14)  
    st.pyplot(plt)

    st.subheader("Summary segment percentages")
    segment_percentages = RFM_data['Segment'].value_counts(normalize=True) * 100
    st.write(segment_percentages)

if submit:
    if uploaded_file is None:
        st.info("Upload a file through config")
        st.stop()

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()

        df = load_data(uploaded_file)
        with st.expander("Data Preview"):
            st.dataframe(df)

        df = CleansingData(uploaded_file)
        RFMmodel(df)



          
# End Main page #