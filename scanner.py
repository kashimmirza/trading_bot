
import yfinance as yf
import config_scanner
from time import sleep
from tda import auth, client
import json
import pandas as pd
import numpy as np

import altair as alt
import seaborn as sns

from trading_bot.agent import Agent
import math
from datetime import datetime, time
import pytz
from pytz import timezone
from time import sleep
import os
import subprocess


import logging
import coloredlogs
from train import main
from eval import main as eval_main
from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
from tda.orders.equities import equity_buy_limit, equity_buy_market, equity_sell_limit, equity_sell_market, equity_buy_to_cover_market, equity_sell_short_market

from trading_bot.methods import evaluate_model
import alpaca_trade_api as tradeapi
from celery import Celery
app = Celery('scanner', broker='redis://localhost:6379/0')

api = tradeapi
run = True


@app.task
def run_scanner():
    ile(
        config_scanner.acc1_token_path, config_scanner.acc1_api_key)
    trading_file = "/home/software34/trading-bot/log.txt"
    signal_file = "/home/software34/trading-bot/signal_log.csv"
    scanner_file = "/home/software34/trading-bot/scanner_log.csv"

    mess = []

    try:
        PUB_KEY = "PKH9J46WWHN3Z4C682PI"
        SEC_KEY = "2oDNfERHI9GNUaF4gR8Hu3QEeCd2TTzh0LQetazx"
        BASE_URL = 'https://paper-api.alpaca.markets'
        api = tradeapi.REST(
            key_id=PUB_KEY, secret_key=SEC_KEY, base_url=BASE_URL)
        r = api.get_account()
    except:
        prclient = auth.client_from_token_fint(
            str("error while connect to alpaca"))


def prepend_line(file_name, line):

    with open(file_name) as f:
        lines = f.readlines()

    with open(file_name, 'w') as f:
        f.write(line)
        f.write("\n")
        for i in lines:
            f.write(str(i))


def in_between(now, start, end):
    if start <= end:
        return start <= now < end
    else:  # over midnight e.g., 23:30-04:15
        return start <= now or now < end


def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions

    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(
        df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)

    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(
            format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )

    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(
            format='$.2f', title='Price'), scale=scale),
        color='action',

        # text = 'action' #--------MY CODES 2 lines
        # ).mark_text(
        #    size = 10
    ).interactive(bind_y=False)

    points2 = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(
            format='$.2f', title='Price'), scale=scale),
        color='action',

        text='action'  # --------MY CODES 2 lines

    ).mark_text(dy=-8,
                size=9
                ).interactive(bind_y=False)

    points3 = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(
            format='$.2f', title='Price'), scale=scale),
        color='action',

        text='date:T'  # --------MY CODES 2 lines
    ).mark_text(dy=10,
                size=8
                ).interactive(bind_y=False)
    # text = alt.Chart(df).encode(
    #    text = 'action'
    # ).mark_text(
    #    size = 10
    # )
    # x = alt.X('max(Date):T'),
    # y = alt.Y('max(actual):Q'),
    # text = alt.Text('max(actual):Q')
    #    text =alt.X('date:T', axis=alt.Axis(title='Date'))
    # )
    # merge the two charts
    merge = points+points2+points3
    chart = alt.layer(actual, merge, title=title).properties(
        height=1000, width=6000)

    return chart


def train(symbol, strategy):

    df_signal = pd.read_csv(signal_file, index_col=0)

    print("\n\n\n- - - - Current Symbol - - -", symbol, " \n")

    # Get the data from Yahoo finance
    train_df = yf.Ticker(symbol).history(period="1Y")
    val_df = yf.Ticker(symbol).history(period="6mo")
    eval_df = yf.Ticker(symbol).history(period="3mo")
    test_df = yf.Ticker(symbol).history(period="1mo")

    # Prepare Data for train val test
    train_df['Adj Close'] = train_df['Close']
    val_df['Adj Close'] = val_df['Close']
    eval_df['Adj Close'] = eval_df['Close']
    test_df['Adj Close'] = test_df['Close']

    # Save Data in the folder
    train_df.to_csv("train_data.csv")
    val_df.to_csv("val_data.csv")
    eval_df.to_csv("eval_data.csv")
    test_df.to_csv("test_data.csv")

    # !python3 train.py train_data.csv val_data.csv --strategy double-dqn --batch-size 2 --episode-count 1
    # os.system("train.py train_data.csv val_data.csv --strategy double-dqn --batch-size 2 --episode-count 1")
    # subprocess.call(['python train.py train_data.csv val_data.csv --strategy double-dqn --batch-size 2 --episode-count 1'])

    # subprocess.Popen("python train.py train_data.csv val_data.csv --strategy double-dqn --batch-size 2 --episode-count 1",
    #                                 cwd=r"/home/software34/trading-bot/",
    #                                 shell=True)

    # train_stock = "/home/software34/trading-bot/train_data.csv"
    # val_stock = "/home/software34/trading-bot/val_data.csv"
    # strategy = config_scanner.strategy
    # window_size = 10
    # batch_size = 2
    # ep_count = 1
    # # model_name = "model_debug_1"
    # model_name = "model_dqn_GOOG_50"

    # # coloredlogs.install(level="DEBUG")
    # switch_k_backend_device()

    # try:
    #     main(train_stock, val_stock, window_size, batch_size,
    #          ep_count, strategy=strategy, model_name=model_name,
    #          pretrained=False, debug=False)
    # except KeyboardInterrupt:
    #     print("Aborted!")

    print("\n\n\n- - - - Training Done - - -", symbol, " \n")

    # !python3 eval.py eval_data.csv --model-name model_debug_1 --debug
    # os.system("eval.py eval_data.csv --model-name model_debug_1 --debug")
    # subprocess.call("eval.py eval_data.csv --model-name model_debug_1 --debug")

    # subprocess.Popen("eval.py eval_data.csv --model-name model_debug_1 --debug",
    #                                 cwd=r"/home/software34/trading-bot/",
    #                                 shell=True)

    eval_stock = "/home/software34/trading-bot/eval_data.csv"
    window_size = 10
    # model_name = 'model_debug_1'
    model_name = strategy
    debug = False

    # coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        eval_main(eval_stock, window_size, model_name, debug)
    except KeyboardInterrupt:
        print("Aborted")

    print("\n\n\n- - - - Eval Done - - -", symbol, " \n")

    # model_name = 'model_debug_1'
    # model_name = "model_dqn_GOOG_50"
    model_name = strategy

    test_stock = "test_data.csv"
    window_size = 10
    debug = True

    agent = Agent(window_size, pretrained=True, model_name=model_name)

    # read csv into dataframe
    df = pd.read_csv(test_stock)
    # filter out the desired features
    df = df[['Date', 'Adj Close']]
    # rename feature column names
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates


#     print(df.head())
    coloredlogs.install(level='DEBUG')
    switch_k_backend_device()

    test_data = get_stock_data(test_stock)
    initial_offset = test_data[1] - test_data[0]

    test_result, history = evaluate_model(agent, test_data, window_size, debug)
    show_eval_result(model_name, test_result, initial_offset)
    chart = visualize(df, history, title=test_stock).interactive()
    chart.save('chart.html')

    # print("\n\n\n- - - - end Done - - -" , symbol , " \n")

    dat = datetime.now(tz=pytz.utc)
    dat = dat.astimezone(timezone('US/Eastern'))
    strategy = config_scanner.strategy

    n = len(df_signal['symbol'])
    df_signal.at[n, 'datetime'] = dat
    df_signal.at[n, 'symbol'] = symbol
    df_signal.at[n, 'signal'] = df.iloc[-1]["action"]
    df_signal.at[n, 'strategy'] = strategy
    df_signal.to_csv(signal_file)

    return (symbol, df.iloc[-1]["action"], test_result)


if __name__ == "__main__":

    df = pd.read_csv(scanner_file, index_col=0)
    dat = datetime.now(tz=pytz.utc)
    dat = dat.astimezone(timezone('US/Eastern'))
    for symbol in config_scanner.symbols:
        n = len(df)
        for strategy in config_scanner.strategy:
            mess = []
            mess.append(str(dat))
            print("\n\n", symbol, strategy, "\n\n")
            profit = train(symbol, strategy)
            profit = profit[2]
            profit = round(profit, 3)
            df.at[n, 'datetime'] = dat
            df.at[n, 'symbol'] = symbol
            df.at[n, str(strategy + "_" + "profit")] = profit
            # df.at[n , 'profit'] = profit
            df.to_csv(scanner_file)
