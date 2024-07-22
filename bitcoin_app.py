import os
import streamlit as st
import plotly_express as px
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import joblib
from keras.models import load_model
import numpy as np
from keras.metrics import MeanSquaredError
from keras.saving import register_keras_serializable
import requests
import time
from datetime import datetime

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_icon=":coin:", layout="wide")

image = Image.open("Bitcoin-Logo.png")
col1, col2 = st.columns([0.1,0.6])
with col1:
    st.image(image,width=150)
html_title = """
    <style>
    .title-test {
    font-weight:bold;
    font-style:Times New Roman;
    padding:7px;
    border-radius:6px;
    }
    </style>
    <h1 class="title-test">CRYPTOCURRENCY PRICE PREDICTION SYSTEM</h1>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)
st.markdown("---------------------------")
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Cryptocurrencies',
                           ['Bitcoin',
                            'Monero',
                            'Solana',
                            'Ethereum',
                            'Get Live Updates'],
                           menu_icon='currency-dollar',
                           icons=['coin','coin', 'coin','coin', 'graph-up-arrow'],
                           default_index=0)
    
def data_visualisation():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig1.update_layout(
        title='Closing Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    st.plotly_chart(fig1)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open'))
    
    fig3.update_layout(
        title='Opening Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    st.plotly_chart(fig3)
    
    fig = px.bar(df, x=df.index, y='Marketcap', title='Market Cap Over Time')
    st.plotly_chart(fig) 
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', yaxis='y1'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['Volume'], mode='lines', name='Volume', yaxis='y2'))
    
    fig2.update_layout(
        title='Price and Volume Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', titlefont=dict(color='red'), tickfont=dict(color='red')),
        yaxis2=dict(title='Volume', titlefont=dict(color='red'), tickfont=dict(color='red'), overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h')
    )
    st.plotly_chart(fig2)
    
    # Create histogram using Plotly
    fig_hist = px.histogram(df, x='Close', nbins=50, title='Price Distribution')
    fig_hist.update_layout(
        xaxis_title='Price',
        yaxis_title='Frequency',
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_hist)
    
    # Create scatter plot using Plotly
    fig_scat = px.scatter(df, x='Close', y='Volume', title='Price vs. Volume Scatterplot', opacity=0.5)
    fig_scat.update_layout(
        xaxis_title='Price',
        yaxis_title='Volume',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scat)
    
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    all_features = ['High', 'Low', 'Open', 'Volume', 'Marketcap']
    
    X = df[all_features]
    y = df['Close']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    # Calculate feature importances
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    # Create a Plotly bar chart
    fig_bar = go.Figure(data=[go.Bar(
        x=[all_features[i] for i in indices],
        y=feature_importances[indices],
        marker_color='blue'  # Change color if desired
    )])
    fig_bar.update_layout(
        title='Feature Importances',
        xaxis_title='Feature',
        yaxis_title='Importance'
    )
    st.plotly_chart(fig_bar)
    # Create a Plotly pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=[all_features[i] for i in range(len(all_features))],
        values=feature_importances,
        hole=0.3  
    )])
    fig_pie.update_layout(
        title='Feature Importances (Pie Chart)'
    )
    st.plotly_chart(fig_pie)

def predict_closing_price():
 
    def predict_price(model, scaler, user_input):
        try:
            # Reshape and scale the input
            user_input_reshaped = user_input.reshape(-1, 1)
            user_input_scaled = scaler.transform(user_input_reshaped).reshape(1, -1, 1)

            # Predict
            prediction = model.predict(user_input_scaled)
            prediction_reshaped = np.repeat(prediction, user_input_scaled.shape[1]).reshape(1, -1)

            # Inverse transform
            prediction_inverse = scaler.inverse_transform(
                np.concatenate((user_input_scaled.reshape(1, -1), prediction_reshaped), axis=0)
            )[1]

            return prediction_inverse[1]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None

    high = st.number_input("Enter the highest price:", min_value=0.0, format="%.5f")
    low = st.number_input("Enter the lowest price:", min_value=0.0, format="%.5f")
    Open = st.number_input("Enter the open price:", min_value=0.0, format="%.5f")
    close = st.number_input("Enter the close price:", min_value=0.0, format="%.5f")
    volume = st.number_input("Enter the volume:", min_value=0.0, format="%.5f")

    if st.button("Predict"):
        user_input = np.array([[high, low, Open, close, volume]])
        predicted_value = predict_price(model, scaler, user_input)
        if predicted_value is not None:
            st.markdown(f'#### Predicted Weighted Price:  {predicted_value}')
            st.markdown("-----------------------------")
            st.markdown("NOTE: The weighted price in cryptocurrency typically refers to the volume-weighted average price (VWAP). This metric provides an average price of a cryptocurrency, taking into account both the price and the volume of trades. VWAP gives a more accurate picture of the average price at which a cryptocurrency has traded over a given period, as it factors in the trading volume at each price point.")
        else:
            st.write('Prediction could not be completed.')

def Live_updates():
    api_key = '0fed01c49681e0274f70c3e49db6dd475dcd0cb67525b135cd3bd72078af317b'
    headers = {
        'Apikey': api_key,
    }

    # Function to fetch cryptocurrency details from CryptoCompare
    def fetch_crypto_details(crypto_symbol):
        api_url = f"https://min-api.cryptocompare.com/data/top/exchanges/full?fsym={crypto_symbol}&tsym=USD"
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None

    # Function to fetch additional data (if available) from CoinGecko
    def fetch_additional_data(crypto_id):
        api_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None

    # Map cryptocurrency symbols to names and CoinGecko IDs
    crypto_symbols = {
        "Bitcoin": "BTC",
        "Ethereum": "ETH",
        "Solana": "SOL",
        "Monero": "XMR"
    }
    crypto_ids = {
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum",
        "Solana": "solana",
        "Monero": "monero"
    }

    # Title of the Streamlit app
    st.markdown("## Live Cryptocurrency Details")

    # Selectbox for choosing cryptocurrency
    crypto = st.selectbox("Choose a cryptocurrency", list(crypto_symbols.keys()))
    st.markdown("---------------------------")
    st.header(f"{crypto} Details:")
    # Placeholder for the live details
    details_placeholder = st.empty()
    
    # Lists to store data for plotting
    time_series = []
    price_series = []

    plot_placeholder = st.empty()

    # Interval for updating the details (in seconds)
    update_interval = 10

    # Fetch and update details in a loop
    while True:
        details = fetch_crypto_details(crypto_symbols[crypto])
        additional_data = fetch_additional_data(crypto_ids[crypto])
        
        if details is not None:
            data = details['Data']['AggregatedData']
            price_usd = data['PRICE']
            market_cap = data['VOLUME24HOURTO']
            volume = data['VOLUME24HOUR']
            total_vol = data.get('TOTALVOLUME24H', 'N/A')
            high_hr = data['HIGH24HOUR']
            total_supply = data.get('SUPPLY', 'N/A')  # Use get to avoid KeyError if SUPPLY is not available    
            
            with details_placeholder:
                left_column1, left_column2, right_column1, right_column2,left_column3, right_column3 = st.columns(6)
                with left_column1:
                    st.subheader("Value in USD:")
                    st.markdown(f"##### US ${price_usd:,.2f}")
                with left_column2:
                    st.subheader("Market Cap:")
                    st.markdown(f"##### US ${market_cap:,.2f}")
                with right_column1:
                    st.subheader("Volume (24hr):")
                    st.markdown(f"##### US ${volume:,.2f}")
                with right_column2:
                    st.subheader("High (24hr):")
                    st.markdown(f"##### US ${high_hr:,.2f}")
                with left_column3:
                    st.subheader("Total Volume (24hr):")
                    st.markdown(f"##### {total_vol if total_vol != 'N/A' else 'Data not available'}")
                with right_column3:
                    st.subheader("Total Supply:")
                    st.markdown(f"##### {total_supply if total_supply != 'N/A' else 'Data not available'}")
               

            
            # Append data to lists for plotting
            time_series.append(datetime.now())
            price_series.append(market_cap)
        
            # Create the plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_series, y=price_series, mode='lines+markers', name='Price (USD)', line=dict(color='red'), marker=dict(color='red')))
            fig.update_layout(title=f'{crypto} Price Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
        
            # Update the plot placeholder
            plot_placeholder.plotly_chart(fig)
        
        else:
            details_placeholder.write("Failed to fetch data.")
        time.sleep(update_interval)
            
            
if selected == "Bitcoin":
    df = pd.read_csv("coin_Bitcoin.csv", encoding='utf-8')
    
    st.title("BITCOIN (BTC)")
    global numeric_columns, date_column
    global non_numeric_columns
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)
    
    st.markdown(""" """)
    image = Image.open("Coin Images/bitcoin.png")
    col1, col2 = st.columns([0.1,0.4])
    with col1:
        st.image(image,width=1070)
    st.markdown(""" """)  
    
    st.subheader("What Is Bitcoin?")
    st.markdown("""Bitcoin (BTC) is a cryptocurrency (a virtual currency) designed to act as money and a form of payment outside the control of any one person, group, or entity.
            This removes the need for trusted third-party involvement (e.g., a mint or bank) in financial transactions.
            It is rewarded to blockchain miners who verify transactions and can be purchased on several exchanges.
            Bitcoin was introduced to the public in 2009 by an anonymous developer or group of developers using the name Satoshi Nakamoto.
          It has since become the most well-known and largest cryptocurrency in the world. Its popularity has inspired the development of many other cryptocurrencies.
          """)
          
    #Visualisation
    st.header("Data Visualizationm of Bitcoin's Data")
    st.write(df)
    legend = """
        ### Dataset Information:
        - **NAME**: Bitcoin
        - **SYMBOL**: BTC
        - **DATE**: 2013-04-30 - 2021-07-07
        - **HIGH RANGES**: 74.6 - 64.9k
        - **LOW RANGES**: 65.5 - 62.2k
        - **OPEN PRICE**: 68.5 - 63.5k
        - **CLOSE PRICE**: 68.4 - 63.5k
        - **VOLUME**: 0 - 351b
        - **MARKETCAP**: 778m - 1186b
        """
    st.markdown(legend)
    
    df=df.drop(['SNo', 'Symbol', 'Name'], axis=1)
    
    data_visualisation() 
    
    st.markdown("------------------------")
    # Register the custom metric
    @register_keras_serializable()
    class CustomMSE(MeanSquaredError):
        pass
    
    model = load_model("models/model_BTC.h5", custom_objects={'mse': CustomMSE})  
    scaler = joblib.load("models/scaler_btc.pkl")
    
    st.header("Price Prediction for BTC")
    predict_closing_price()
     
if selected == "Monero":
    df = pd.read_csv("coin_Monero.csv")
    
    st.title("MONERO (XMR)")
    global numeric_columns, date_column
    global non_numeric_columns
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)
    
    st.markdown(""" """)
    image = Image.open("Coin Images/monero.jpg")
    col1, col2 = st.columns([0.1,0.5])
    with col1:
        st.image(image,width=1070)
    st.markdown(""" """)  
    
    st.subheader("What Is Monero?")
    st.markdown("""Monero (XMR) is the leading cryptocurrency focused on private and censorship-resistant transactions.
                The majority of existing cryptocurrencies, including Bitcoin and Ethereum, have transparent blockchains. 
                Transactions can be verified and/or traced by anyone in the world. This means that the sending and receiving addresses of these transactions could potentially be linked to real-world identities.
                Monero, on the other hand, uses various technologies to ensure the privacy of its users. Monero transactions are confidential and untraceable.
          """)
          
    #Visualisation
    st.header("Data Visualizationm of Monero's Data")
    st.write(df)
    legend = """
        ### Dataset Information:
        - **NAME**: Monero
        - **SYMBOL**: XMR
        - **DATE**: 2014-05-23 - 2021-07-07
        - **HIGH RANGES**: 0.25 - 518k
        - **LOW RANGES**: 0.21 - 453k
        - **OPEN PRICE**: 0.22 - 48.57k
        - **CLOSE PRICE**: 0.22 - 48.57k
        - **VOLUME**: 7.9k - 29b
        - **MARKETCAP**: 1.28m - 8.66b
        """
    st.markdown(legend)
    
    df=df.drop(['SNo', 'Symbol', 'Name'], axis=1)
    
    data_visualisation()
    
    st.markdown("------------------------")
    # Register the custom metric
    @register_keras_serializable()
    class CustomMSE(MeanSquaredError):
        pass
    
    model = load_model("models/model_XMR.h5", custom_objects={'mse': CustomMSE})  
    scaler = joblib.load("models/scaler_xmr.pkl")
    
    st.header("Price Prediction for XMR")
    predict_closing_price()
    

if selected == "Solana":
    df = pd.read_csv("coin_Solana.csv")
    
    st.title("SOLANA (SOL)")
    global numeric_columns, date_column
    global non_numeric_columns
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)
    
    st.markdown(""" """)
    image = Image.open("Coin Images/solana.png")
    col1, col2 = st.columns([0.1,0.5])
    with col1:
        st.image(image,width=1070)
    st.markdown(""" """)  
    
    st.subheader("What Is Solana?")
    st.markdown("""Solana (SOL) is an open infrastructure for building scalable crypto apps. 
                The architecture is censorship resistant, fast, and secure, and designed to facilitate global adoption. 
                To keep time on the blockchain, Solana employs an innovative process known as Proof of History.
                Solana is designed to provide high transaction speeds, making it well-suited for applications that require fast processing.
                """)
          
    #Visualisation
    st.header("Data Visualizationm of Solana's Data")
    st.write(df)
    legend = """
        ### Dataset Information:
        - **NAME**: Solana
        - **SYMBOL**: SOL
        - **DATE**: 2020-04-12 - 2021-07-07
        - **HIGH RANGES**: 0.56 - 58.3
        - **LOW RANGES**: 0.51 - 46.2
        - **OPEN PRICE**: 0.51 - 56.1
        - **CLOSE PRICE**: 0.52 - 55.9
        - **VOLUME**: 652k - 2.77b
        - **MARKETCAP**: 0 - 15.2b
        """
    st.markdown(legend)
    
    df=df.drop(['SNo', 'Symbol', 'Name'], axis=1)
    
    data_visualisation()
    
    st.markdown("------------------------")
    # Register the custom metric
    @register_keras_serializable()
    class CustomMSE(MeanSquaredError):
        pass
    
    model = load_model("models/model_SOL.h5", custom_objects={'mse': CustomMSE})  
    scaler = joblib.load("models/scaler_sol.pkl")
    
    st.header("Price Prediction for SOL")
    predict_closing_price()
    
if selected == "Ethereum":
    df = pd.read_csv("coin_Ethereum.csv")
    
    st.title("ETHEREUM (ETH)")
    global numeric_columns, date_column
    global non_numeric_columns
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)
    
    st.markdown(""" """)
    image = Image.open("Coin Images/Ethereum.jpg")
    col1, col2 = st.columns([0.1,0.5])
    with col1:
        st.image(image,width=1070)
    st.markdown(""" """)  
    
    st.subheader("What Is Ethereum?")
    st.markdown("""Ethereum (ETH) is a decentralized blockchain and development platform. It allows developers to build and deploy applications and smart contracts. 
                Ethereum utilizes its native cryptocurrency, ether (ETH), for transactions and incentivizes network participants through proof-of-stake (PoS) validation.
                It has a large and committed global community and the largest ecosystem in blockchain and cryptocurrency. 
                Wide range of functions. Besides being used as a digital currency, Ethereum can also process other financial transactions, execute smart contracts and store data for third-party applications.
                """)
          
    #Visualisation
    st.header("Data Visualizationm of Ethereum's Data")
    st.write(df)
    legend = """
        ### Dataset Information:
        - **NAME**: Ethereum
        - **SYMBOL**: ETH
        - **DATE**: 2015-08-09 - 2021-07-07
        - **HIGH RANGES**: 0.48 - 4.36k
        - **LOW RANGES**: 0.42 - 3.79k
        - **OPEN PRICE**: 0.43 - 4.17k
        - **CLOSE PRICE**: 0.43 - 4.17k
        - **VOLUME**: 102k - 84.5b
        - **MARKETCAP**: 32.2m - 483b
        """
    st.markdown(legend)
    
    df=df.drop(['SNo', 'Symbol', 'Name'], axis=1)
    
    data_visualisation()
    
    st.markdown("------------------------")
    # Register the custom metric
    @register_keras_serializable()
    class CustomMSE(MeanSquaredError):
        pass
    
    model = load_model("models/model_ETH.h5", custom_objects={'mse': CustomMSE})  
    scaler = joblib.load("models/scaler_eth.pkl")
    
    st.header("Price Prediction for ETH")
    predict_closing_price()
    

if selected == "Get Live Updates":
    Live_updates()

    
