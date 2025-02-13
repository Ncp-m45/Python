import streamlit as st
import pandas as pd
import numpy as np
import random
from io import StringIO
import squarify
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px

### Page setup ###
st.set_page_config(page_title="Analysis Dashboard", page_icon=":bar_chart:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>',unsafe_allow_html=True) #ปรับ top padding
st.title("Data analysis Dashbord")
# End Page setup #

with st.sidebar:
    # Upload file #
    @st.cache_data
    def load_data(file):
            data = pd.read_csv(file)
            return data

    uploaded_file = st.file_uploader("Choose a file")

    # Cleansing Data #
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
    
    submit = st.button("SUBMIT", type="primary")
# End Sidebar #


### Definition Convert RFM Score to segment label ###
def segment_label(RFMScore):
    if RFMScore in ['54', '55']:
        return "Champion"
    elif RFMScore == '52':
        return "Recent User"
    elif RFMScore == '51':
        return "Price Sensitive"
    elif RFMScore in ['42', '43', '52', '53']:
        return "Potential Loyalist"
    elif RFMScore == '41':
        return "Promising"
    elif RFMScore in ['34', '35', '44', '45']:
        return "Loyal Customer"
    elif RFMScore == '33':
        return "Needs Attention"
    elif RFMScore in ['31', '32']:
        return "About to Sleep"
    elif RFMScore in ['15', '25']:
        return "Can't Lose Them"
    elif RFMScore in ['13', '14', '23', '24']:
        return "Hibernating"
    elif RFMScore in ['11', '12', '21', '22']:
        return "Lost"
    else:
        return "Don't have segment label"
# End def #
    
### Definition create RFM model ###    
def RFMmodel(df):
    RFM_data = pd.concat([df['recency'], df['frequency'],df['monetary']], axis=1)
    RFM_data['RecencyScore'] = pd.qcut(RFM_data['recency'], 5, labels=[5, 4, 3, 2, 1])
    RFM_data['FrequencyScore'] = pd.qcut(RFM_data['frequency'].rank(method = 'first'),5, labels = [1, 2, 3, 4, 5])
    RFM_data['MonetaryScore'] = pd.qcut(RFM_data['monetary'], 5, labels = [1, 2, 3, 4, 5])
    RFM_data['FMScore'] = (RFM_data['FrequencyScore'].astype('float') + RFM_data['MonetaryScore'].astype('float'))/2
    RFM_data['FMScore'] = np.ceil(RFM_data['FMScore']).astype('int')
    RFM_data['RFMScore'] = RFM_data['RecencyScore'].astype('str') + RFM_data['FMScore'].astype('str') 

    RFM_data['Segment'] = RFM_data['RFMScore'].apply(segment_label)

    # Calculate average values for each RFM_Segment_Label
    segment_summary = RFM_data.groupby('Segment').agg(
        Recency_Avg=('recency', 'mean'),
        Frequency_Avg=('frequency', 'mean'),
        Monetary_Avg=('monetary', 'mean'),
        Segment_Size=('Segment', 'count')
    ).reset_index()
    
    result = RFM_data.groupby(['Segment'])['Segment'].count()
    values = list(result)
    labels = result.index
    total_customers = len(RFM_data)
    labels_with_percentage = [f"{label} ({value / total_customers * 100:.2f}%)" for label, value in zip(labels, values)]

    # Plot model
    colors = ['#070F2B', '#1B1A55', '#535C91', '#9290C3']
    plt.figure(figsize = (18, 11), facecolor='none')  
    squarify.plot(sizes=values, color=colors, label=labels_with_percentage ,text_kwargs={'color': 'white', 'fontsize': 12})
    plt.title('Customer segmentation',fontsize=16)
    plt.xlabel('Recency',color='white', fontsize=16)
    plt.ylabel('FMScore',color='white', fontsize=16)
    plt.tick_params(axis='x', colors='white', labelsize=14)  
    plt.tick_params(axis='y', colors='white', labelsize=14)  
    st.pyplot(plt)

    return segment_summary
# End def #

### Definition plot metric ###
def plot_metric(label, value=0.00, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value= value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )
    st.plotly_chart(fig, use_container_width=True)
# End def #

### Definition bar chart ###
def bar_chart(df, x, y, lable, title):
    fig = px.bar(
        df,
        x= x,
        y= y,
        color=lable,
        barmode="group",
        text_auto=".2s",
        title=title
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    st.plotly_chart(fig, use_container_width=True) 
# End def #

### Main layout ###
if submit:
    if uploaded_file is None:
        st.info("Upload a file through config")
        st.stop()

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        df = load_data(uploaded_file)

    if uploaded_file.name == 'Data_sample.csv':
        RFMmodel(df)

    if uploaded_file.name == 'OnlineRetail.csv':
        tab1, tab2 = st.tabs(['Dashbord', 'Summarizing'])
    ### Dashbord ###        
        with tab1:
            cleaned_data = CleansingData(uploaded_file)
            RFMmodel(cleaned_data)

            if uploaded_file.name == 'OnlineRetail.csv':
                df = df.dropna()
                df = df.drop_duplicates()

                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                max_year = max(df['InvoiceDate'].dt.year.unique())

                df['TotalSales'] = df['Quantity'] * df['UnitPrice']
                canceled_products = df[df['InvoiceNo'].str.contains('C', na=False)]

                c1, c2, c3 = st.columns(3)

                with c1:
                    plot_metric(
                        f"Total sales {max_year}",
                        df[df['InvoiceDate'].dt.year == max_year]['TotalSales'].sum(),
                        prefix="$",
                        suffix="",
                        show_graph=True,
                        color_graph="rgba(0, 104, 201, 0.2)"
                    )
                with c2:
                    plot_metric(f"Total called products {max_year}", 
                                df[(df['InvoiceDate'].dt.year == max_year) & (df['InvoiceNo'].str.contains('C', na=False))]['TotalSales'].sum()*(-1),
                                prefix="$", suffix="", show_graph=True ,color_graph="rgba(0, 104, 201, 0.2)")

                with c3:
                    plot_metric("Total number of members", 
                                len(df['CustomerID'].value_counts()),
                                prefix="", suffix="", show_graph=True ,color_graph="rgba(0, 104, 201, 0.2)")
                    
                # Graph comparing total sales vs canceled sales
                totalSales_canceled = canceled_products['TotalSales'].sum()
                totalSales_non_canceled = df['TotalSales'].sum()

                # Create DataFrame for Comparing
                sales_comparison = pd.DataFrame({
                    'Status': ['Non-Canceled', 'Canceled'],
                    'Total Sales': [totalSales_non_canceled, totalSales_canceled]
                })
                fig_bar1 = px.bar(sales_comparison, x='Status',y='Total Sales',text='Total Sales',
                                title="Comparison of Total Sales: Non-Canceled vs Canceled Orders",
                                color='Status',
                                color_discrete_map={'Non-Canceled': 'green', 'Canceled': 'red'}
                )
                st.plotly_chart(fig_bar1, use_container_width=True)

                cl1 , cl2 = st.columns(2)
                with cl1:
                    df = df[~df.isin(canceled_products)].dropna()
                    with st.expander("Non-Canceled Orders"):
                        st.write(f"*Number of orders by members: {len(df):,}*")
                        st.write(df)

                with cl2:
                    with st.expander("Canceled Orders"):
                        st.write(f"Number of products that were canceled: {len(canceled_products):,}")
                        st.write(canceled_products)

                
                filtered_df = df.copy()
                filtered_df['TotalSales'] = df['Quantity'] * df['UnitPrice']
                filtered_df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                filtered_df['Date'] = filtered_df['InvoiceDate'].dt.strftime("%Y-%m-%d")
                filtered_df['Month'] = filtered_df['InvoiceDate'].dt.to_period('M').astype(str)

                # Calculate total daily sales
                daily_sales = filtered_df.groupby('Date', as_index=False)['TotalSales'].sum()

                # Calculate monthly total sales
                monthly_sales = filtered_df.groupby('Month', as_index=False)['TotalSales'].sum()

                a1, a2 = st.columns(2)
                with a1:
                    temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
                    temp = temp.reset_index(drop = False)
                    countries = temp['Country'].value_counts()

                    # Define the data for the choropleth map
                    data = dict(
                        type='choropleth',
                        locations=countries.index,
                        locationmode='country names',
                        z=countries,
                        text=countries.index,
                        colorbar={'title': 'Order nb.'},
                        colorscale=[
                                [0, 'rgb(230,230,250)'],
                                [0.01, 'rgb(166,206,227)'],
                                [0.02, 'rgb(31,120,180)'],
                                [0.03, 'rgb(178,223,138)'],
                                [0.05, 'rgb(51,160,44)'],
                                [0.10, 'rgb(251,154,153)'],
                                [0.20, 'rgb(255,255,0)'],
                                [1, 'rgb(227,26,28)']
                        ],
                        reversescale=False
                    )
                
                    layout = dict(
                        title='Number of Orders per Country',
                        geo=dict(showframe=True, projection={'type': 'mercator'})
                    )
                    choromap = go.Figure(data=[data], layout=layout)
                    st.plotly_chart(choromap)

                # Country with Sales
                with a2:
                    st.subheader("Country with Sales")
                    fig_pie = px.pie(filtered_df, values = filtered_df['TotalSales'] , names = "Country")
                    fig_pie.update_traces(text = filtered_df["Country"] , textposition = "inside")
                    st.plotly_chart(fig_pie , use_container_width = True , height = 650)

                ### Top 5 ###
                # Select the top 5 of Quantity
                filtered_df_product = df.groupby(by=['StockCode', 'Description'], as_index=False).agg({'Quantity': 'sum','TotalSales': 'sum','StockCode' : 'count'})
                filtered_df_product = filtered_df_product.rename(columns = {'Quantity': 'Total Quantity','TotalSales': 'Total Sales per Product','StockCode' : 'Total orders per product'})
                top_5_products = filtered_df_product.nlargest(5, 'Total Quantity')

                # graph
                fig_pie2 = px.pie(top_5_products, values = top_5_products['Total Quantity'] , names = 'Description',
                                title = "Top 5 Products by Total Quantity")
                fig_pie2.update_traces(text = filtered_df['Description'] , textposition = "outside")
                st.plotly_chart(fig_pie2)
            
                # Select the top 5 
                top_5_products = filtered_df_product.nlargest(5, 'Total Sales per Product')

                # graph
                fig_pie2 = px.pie(top_5_products, values = top_5_products['Total Sales per Product'] , names = 'Description',
                                title = "Top 5 Products by Total Sales per Product",template="gridon")
                fig_pie2.update_traces(text = filtered_df['Description'] , textposition = "outside")
                st.plotly_chart(fig_pie2)

                # Select the top 5
                top_5_products = filtered_df_product.nlargest(5, 'Total orders per product')

                # graph
                fig_pie2 = px.pie(top_5_products, values = top_5_products['Total orders per product'] , names = 'Description',
                                title = "Top 5 Products by Total Orders per Product", template='plotly_dark')
                fig_pie2.update_traces(text = filtered_df['Description'] , textposition = "outside")
                st.plotly_chart(fig_pie2)
                # End Top 5 #

                b1, b2 = st.columns(2)

                # Weekly Sales
                with b1:
                    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
                    df['TotalSales'] = df['Quantity'] * df['UnitPrice']

                    sales_by_day = df.groupby(['DayOfWeek'])['TotalSales'].sum()
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    sales_by_day = sales_by_day.reindex(days)
                    sales_by_day = sales_by_day.reset_index()
                    sales_by_day.columns = ['Day of Week', 'Total Sales']

                    bar_chart(sales_by_day, 'Day of Week', 'Total Sales', 'Day of Week', 'Weekly Sales by Invoice Date')

                # Sales by Time Period
                with b2:
                    # Create a function to find a time period
                    def time_of_day(hour):
                        if 6 <= hour < 12:
                            return 'Morning'
                        elif 12 <= hour < 18:
                            return 'Afternoon'
                        elif 18 <= hour < 24:
                            return 'Evening'
                        else:
                            return 'Night'

                    # Calculate total sales by time period
                    filtered_df['TimePeriod'] = filtered_df['InvoiceDate'].dt.hour.apply(time_of_day)
                    time_period_sales = filtered_df.groupby('TimePeriod' , as_index=False)['TotalSales'].sum()
                    fig_bar1 = px.bar(time_period_sales, x='TimePeriod', y='TotalSales', color='TimePeriod', title='Sales by Time Period')
                    st.plotly_chart(fig_bar1, use_container_width=True)

                cl1 , cl2 = st.columns(2)

                # Daily Sales
                with cl1:
                    fig_line1 = px.line(daily_sales, x='Date', y='TotalSales', title='Daily Sales',
                                        labels={"TotalSales": "Amount"}, height=500, width=1000,
                                        template="gridon")
                    st.plotly_chart(fig_line1, use_container_width=True)
                
                # Monthly Sales
                with cl2:
                    fig_line2 = px.line(monthly_sales, x='Month', y='TotalSales', title='Monthly Sales',
                                        labels={"TotalSales": "Amount"}, height=500, width=1000,
                                        template="gridon")
                    st.plotly_chart(fig_line2, use_container_width=True) 

### Summarizing the results ###
        with tab2:
            with st.expander("Data Preview"):
                st.markdown(f"Number of data: {len(load_data(uploaded_file)):,}")
                variables = '''**This dataframe contains 8 variables that correspond to:**  
    **InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.  
    **StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.  
    **Description**: Product (item) name. Nominal.  
    **Quantity**: The quantities of each product (item) per transaction. Numeric.  
    **InvoiceDate**: Invice Date and time. Numeric, the day and time when each transaction was generated.  
    **UnitPrice**: Unit price. Numeric, Product price per unit in sterling.  
    **CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.  
    **Country**: Country name. Nominal, the name of the country where each customer resides.
    '''
                st.markdown(variables)
                st.dataframe(load_data(uploaded_file))

            cleaned_data = CleansingData(uploaded_file)
            with st.expander("Data for RFM model"):
                st.markdown(f"Number of data: {len(cleaned_data):,}")
                c1, c2 = st.columns(2)
                with c1:
                    st.write(cleaned_data)               
                    csv = cleaned_data.to_csv().encode("utf-8")
                    st.download_button(
                        label="Download cleaned data as CSV",
                        data=csv,
                        file_name="RFM of Online Retail.csv",
                        mime="text/csv",
                    )
                with c2:
                    segment_summary = RFMmodel(cleaned_data)

            st.write(segment_summary)

            ### Summarized Results ###
            with st.expander('Insights of Customer behavior'):  
                st.markdown('''จะเห็นได้ว่ามีลูกค้าเพียงประมาณ 45% ที่อยู่ในระดับ RFM สูงสุด ร้านค้าจะต้องพยายามรักษาความภักดีนี้ไว้ และต้องกระตุ้นลูกค้าในส่วนที่เหลือให้ได้มากที่สุด
                ซึ่งลูกค้าส่วนใหญ่จะอยู่ที่ Lost, Loyal Customer, Hibernating และ Champion ตามลำดับ โดยทางเราได้ทำการวิเคราะห์กลยุทธ์ทางตลาด ดังนี้  
                **Champion** (15.30%): ลูกค้ากลุ่มนี้มีความมั่งคั่งทางการเงินค่อนข้างสูง ค่าใช้จ่ายเฉลี่ยอยู่ที่ 442เหรียญ/ครั้ง คาดว่าส่วนใหญ่เป็นนักธุรกิจและมีความจำเป็นต้องใช้สินค้าของทางร้านบ่อยๆ ทางร้านค้าควรที่จะคอยหมั่นเสนอสิทธิพิเศษให้แก่ลูกค้ากลุ่มนี้ เช่น สิทธิ์ในการสั่งซื้อสินค้าล่วงหน้าและเข้าถึงสินค้าได้ก่อนใคร ส่วนลดสำหรับสมาชิกระดับสูง หรือ กิจกรรมพิเศษ  
                **Loyal Customer** (20.36%): ขาชอปประจำร้าน เฉลี่ยซื้อครั้งละ 371เหรียญ/ครั้ง ทางร้านค้าควรรักษาการติดต่อระหว่างลูกค้ากลุ่มนี้ไว้ ส่งสิทธิพิเศษ กิจกรรมสะสมแต้มจากยอดสั่งซื้อ หรือ โปรโมชั่นสินค้าที่ลูกค้ากลุ่มนี้ซื้อบ่อยๆ เพื่อกระตุ้นให้พวกเขาใช้จ่ายอยู่เสมอ  
                **Potential Loyalist** (9.79%): ลูกค้ากลุ่มนี้อาจจะเปลี่ยนตัวเลือกร้านค้าเป็นร้านของคุณแทน หากพวกเขาชื่นชอบสินค้าลดราคา หรือมีการลดแลกแจกแถมเป็นพิเศษ ทางร้านควรแจกของสมนาคุณในการซื้อครั้งต่อไป เสนอส่วนลดพิเศษสำหรับสมาชิก หรือ โปรโมชั่นส่งฟรีหากยอดชำระถึงเกณฑ์ที่กำหนดไว้ เพื่อแสดงว่าทางร้านให้ความสำคัญกับพวกเขา  
                **Needs Attention** (4.69%): ลูกค้ากลุ่มนี้ห่างหายจากร้านไปค่อนข้างนาน แต่มีการซื้อเฉลี่ยต่อครั้งค่อนข้างสูง อยู่ที่ประมาณ 300เหรียญ/ครั้ง ทางร้านค้าควรติดต่อสอบถามข้อมูลและหาแนวทางปรับแก้ไขเพื่อให้ลูกค้ากลับมาใช้บริการอีกครั้ง อาจจัดโปรโมชั่นต้อนรับสมาชิกเก่าที่ห่างหายไปนาน หรือ เพิ่มเติมสินค้าให้ตรงกับความต้องการของลูกค้า และอาจจะสร้างการรับรู้ถึง Brands ให้มากยิ่งขึ้น  bbbbbbbb
                **Hibernating** (15.65%): แม้ว่าลูกค้ากลุ่มนี้จะขาดการสั่งซื้อไปนานแล้ว แต่หากสามารถเชิญชวนกลับมาได้จะช่วยให้ยอดขายเพิ่มขึ้นค่อนข้างสูง เนื่องจากลูกค้ากลุ่มนี้สั่งซื้อแต่ละครั้งประมาณ 428เหรียญ ทางร้านควรติดต่อสอบถามลูกค้าและหาแนวทางแก้ไขเช่นเดียวกับลูกค้ากลุ่ม Needs Attention คอยส่งโปรโมชั่นเกี่ยวกับสินค้าที่ลูกค้าสนใจพร้อมกับบริการส่งฟรีโดยกำหนดระยะเวลาเพื่อกระตุ้นให้ลูกค้าใช้จ่ายทันที และจัดกิจกรรมแจกของรางวัลหรือส่วนลดเพื่อต้อนรับการกลับมา  
                '''
                )  

            filter = load_data(uploaded_file)
            filter = filter.dropna()
            # Summary Data (Data)
            df_summary = pd.DataFrame([{'Products': len(filter['StockCode'].value_counts()),  
            'Canceled_products': len(canceled_products),  
            'Transactions': len(filter['InvoiceNo'].value_counts()),
            'Customers': len(filter['CustomerID'].value_counts()),
            'Countries': len(filter['Country'].value_counts())
            }], columns = ['Products', 'Canceled_products', 'Transactions', 'Customers', 'Countries'], index = ['Quantity'])
            st.dataframe(df_summary)

            # Customer Invoice Summary
            st.subheader("Customer Invoice Summary")
            df_productCount = df.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False).agg({'InvoiceDate': 'count','Quantity': 'sum'})
            df_productCount = df_productCount.rename(columns={'InvoiceDate': 'List Product per Invoice','Quantity': 'Total Quantity Product'})
            df_productCount[:10].sort_values('CustomerID')
            st.dataframe(df_productCount)

            # Product Sales Summary
            st.subheader("Product Sales Summary")
            df_product = df.groupby(by=['StockCode','Description'], as_index=False).agg({'Quantity': 'sum','TotalSales': 'sum','StockCode' : 'count'})
            df_product = df_product.rename(columns = {'Quantity': 'Total Quantity','TotalSales': 'TotalSales per Product','StockCode' : 'Total orders per product'})
            st.dataframe(df_product, width=1000)



                
