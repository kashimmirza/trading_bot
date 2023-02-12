
import yfinance as yf
import config
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
from celery import Celery
from celery.schedules import crontab


import logging
import coloredlogs
from train import main
from eval import main as eval_main
from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
from tda.orders.equities import equity_buy_limit, equity_buy_market, equity_sell_limit, equity_sell_market, equity_buy_to_cover_market, equity_sell_short_market

from trading_bot.methods import evaluate_model
import alpaca_trade_api as tradeapi


api = tradeapi
run = True


client = auth.client_from_token_file(
    config.acc1_token_path, config.acc1_api_key)
trading_file = "/home/software34/trading-bot/log.txt"
signal_file = "/home/software34/trading-bot/signal_log.csv"
mess = []


try:
    PUB_KEY = "PKH9J46WWHN3Z4C682PI"
    SEC_KEY = "2oDNfERHI9GNUaF4gR8Hu3QEeCd2TTzh0LQetazx"
    BASE_URL = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(key_id=PUB_KEY, secret_key=SEC_KEY, base_url=BASE_URL)
    r = api.get_account()
except:
    print(str("error while connect to alpaca"))


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


@app.task
def buy_order(stock, amount, limit_price):
    open_position = []
    res = client.get_account(account_id=config.acc_id,
                             fields=client.Account.Fields.POSITIONS).json()
    pos = res['securitiesAccount']["positions"]
    for i in range(len(pos)):
        open_position.append(pos[i]['instrument']['symbol'])

    if stock in open_position:
        mess.append(str("\nThe Symbol : " + stock + " Already in Position"))
    else:

        res = client.get_accounts()
        cash = res.json()[0]
        bp = float(cash["securitiesAccount"]["currentBalances"]['buyingPower'])

        if amount < bp:
            try:
                res = client.get_quote(stock)
                cp = float(res.json()[stock]["closePrice"])

                quantity = amount / cp
                quantity = round(quantity, 2)
                price = round(cp, 2)
                total_price = price * quantity
                total_price = round(total_price, 2)

                quantity = math.floor(quantity)
                if quantity == 0:
                    quantity = 1

                # Place Market Buy Order
                # res = client.place_order(
                #     config.acc_id,  # account_id
                #     equity_buy_market(stock , quantity = quantity )
                # )

                # Place Limit Buy Order
                res = client.place_order(
                    config.acc_id,  # account_id
                    equity_buy_limit(
                        symbol=stock, quantity=quantity, price=limit_price)
                )

                qtyy = str(quantity)
                price = str(limit_price)
                total_price = str(total_price)
                symbo = str(stock)

                date_format = '%m/%d/%Y %H:%M:%S'
                dat = datetime.now(tz=pytz.utc)
                dat = dat.astimezone(timezone('US/Pacific'))
                dat = dat.strftime(date_format)
                # prin = (dat + "\t Market Buy " + symbo + " With " + qtyy + " Shares and amount $" + total_price + " at price $"  + price)
                prin = (dat + "\t Limit Buy " + symbo + " With " + qtyy +
                        " Shares and amount $" + total_price + " at limit price $" + price)

                prepend_line(trading_file, prin)
                mess.append(str(prin))

                mess.append(str("\n"))
                mess.append(str(res))

            except Exception as e:
                prin = str("Buy Order Cannot Place for Stock " +
                           str(stock) + " Error " + str(e))
                prepend_line(trading_file, prin)
                mess.append(prin)
        else:
            mess.append(str("\n - - Low Cash Cannot Buy - - " + stock))
            prin = str("Buy Order Cannot Place for Stock " +
                       str(stock) + " Low buying Power ")
            prepend_line(trading_file, prin)
            mess.append(prin)


@app.task
def sell_order(stock):
    try:
        open_position = []
        res = client.get_account(
            account_id=config.acc_id, fields=client.Account.Fields.POSITIONS).json()
        pos = res['securitiesAccount']["positions"]
        for i in range(len(pos)):
            open_position.append(pos[i]['instrument']['symbol'])

        if stock in open_position:

            for j in range(len(pos)):
                if stock == pos[j]['instrument']['symbol']:
                    qty = int(pos[j]['longQuantity'])
                    pl = pos[j]['currentDayProfitLossPercentage']

            # Place Sell Order
            res = client.place_order(
                config.acc_id,  # account_id
                equity_sell_market(symbol=stock, quantity=qty)
            )

            mess.append(str(res))

            res = client.get_quote(stock)
            cp = float(res.json()[stock]["closePrice"])

            date_format = '%m/%d/%Y %H:%M:%S'
            dat = datetime.now(tz=pytz.utc)
            dat = dat.astimezone(timezone('US/Pacific'))
            dat = dat.strftime(date_format)
            prin = str(dat + "\t Market SELL " + stock + " With " + str(qty) +
                       " Shares at price $" + str(cp) + " Day ProfitLossPercentage " + str(pl) + ".")
            prepend_line(trading_file, prin)
            mess.append(str("\n"))
            mess.append(prin)
        else:
            mess.append(str("\nThe Symbol : " + stock + " is not in Position"))
            prin = str("\nThe Symbol : " + stock + " is not in Position")
            prepend_line(trading_file, prin)
    except Exception as e:
        prin = str("Sell Order Cannot Place for Stock " +
                   str(stock) + " Error " + str(e))
        prepend_line(trading_file, prin)
        mess.append(prin)


# @app.task
def train(symbol):

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
    # strategy = config.strategy
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
    model_name = config.strategy
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
    model_name = config.strategy

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
    strategy = config.strategy

    n = len(df_signal['symbol'])
    df_signal.at[n, 'datetime'] = dat
    df_signal.at[n, 'symbol'] = symbol
    df_signal.at[n, 'signal'] = df.iloc[-1]["action"]
    df_signal.at[n, 'strategy'] = strategy
    df_signal.to_csv(signal_file)

    return (symbol, df.iloc[-1]["action"], history[-1][0])

    # while True:


@shared_task(bind=True)
# @app.task
def trade_task():
    dat = datetime.now(tz=pytz.utc)
    dat = dat.astimezone(timezone('US/Eastern'))
    # if in_between(dat.time() ,  time(9), time(10)) and run and api.get_clock().is_open:
    if run and api.get_clock().is_open:
        for symbol in config.symbols:
            mess = []
            mess.append(str(dat))
            symbol, message, limit_price = train(symbol)
            limit_price = round(limit_price, 2)
            print(symbol, message, limit_price)
            if message == "BUY":
                buy_order(stock=symbol, amount=config.amount,
                          limit_price=limit_price)
            elif message == "SELL":
                sell_order(stock=symbol)
            else:
                print("Hold")
            print(mess)
        run = False
    else:
        print("Not a good Time to trade")
        sleep((api.get_clock().next_open.astimezone(
            tz=pytz.timezone("US/Eastern")) - dat).seconds)
