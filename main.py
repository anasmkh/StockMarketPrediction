import pandas as pd
import plotly.express as px
from prophet import Prophet
import streamlit as st
from PIL import Image

image = Image.open('stock.jpg')
st.set_page_config(
    page_title="Stock Price Prediction Tool",
    page_icon=image,
)
st.sidebar.title("Stock Prediction Tool")

st.title("Stock Price Prediction Service")
st.sidebar.image('stock.jpg')
st.sidebar.markdown('easy and powerful tool ')
st.sidebar.title('Upload your data file ')
uploaded_file = st.sidebar.file_uploader('')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st1, st2, st3 = st.columns([1, 0.2, 0.5])
    period = int(st.sidebar.slider('Enter the prediction period /per day', 1, 60))
    ####################################################################
    st.sidebar.title("Process Current Data")

    with st.container():
        if st.sidebar.checkbox('show data description'):
            st.subheader('Your Row Data')
            st.write(df)
            if st.button('show data description'):
                st.subheader('Data Description')
                st.write(df.describe())

    if st.sidebar.checkbox("plot data bar chart"):
        st.subheader('Data over time chart')
        st.write(px.bar(df, x='Date', y='Close'))
    if st.sidebar.checkbox('plot data line chart'):
        st.subheader('Data over time chart')
        st.write(px.line(df, x='Date', y='Close', title='Close to Date Chart'))

    columns = ['Date', 'Close']

    df_new = pd.DataFrame(df, columns=columns)

    df_new = df_new.rename(columns={'Date': 'ds', 'Close': 'y'})

    m = Prophet()

    m.fit(df_new)
    ##########################

    future = m.make_future_dataframe(periods=period)
    forCast = m.predict(future)
    st.sidebar.title("Make Predictions")
    if st.sidebar.checkbox("plot predictions"):
        st.header('New predictions')
        st.write(px.area(forCast, x='ds', y='yhat', title='Predictions are based on the last year'))
        st.write(px.line(forCast, x='ds', y='yhat', title='line chart'))
        st.write(px.bar(forCast, x='ds', y='yhat', title='Bar Chart'))

    weekly = st.sidebar.button('see more detailed prediction ', key='weekly')
    if weekly:
        st.write(m.plot(forCast, xlabel='date', ylabel='price', figsize=(10, 10)))
        st.write(m.plot_components(forCast, weekly_start=1, yearly_start=2022))


    def convert_df(for_cast):

        return for_cast.to_csv().encode('utf-8')


    csv = convert_df(forCast)
    st.download_button(
        label="Download predictions file",
        data=csv,
        file_name='predictions_file.csv',
        mime='text/csv'
    )
