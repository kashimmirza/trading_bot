from django.models
import math
from django.shortcuts import render
# Create your views here.
import os
import logging
from django.shortcuts import redirect, render, HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
from .models import TradingAccount

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
# from .ops import (
#     get_state
# )
from django.shortcuts import render, redirect
from .forms import TradingAccountForm


# @login_required(login_url="signin")

def create_trading_account(request):
    if request.method == 'POST':
        form = TradingAccountForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('trading_account_list')
    else:
        form = TradingAccountForm()
    return render(request, 'trading/create_trading_account.html', {'form': form})


def trading_account_list(request):
    trading_accounts = TradingAccount.objects.all()
    return render(request, 'trading/trading_account_list.html', {'trading_accounts': trading_accounts})


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta  # max(delta, 0)
            total_profit += delta

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # if episode % 10 == 0:
    agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []

    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta  # max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * \
        [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


def signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        if User.objects.filter(username=username).exists():
            messages.warning(request, "User Already Exists!")
            return render(request, "signin.html")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")
        if password1 == password2:
            myuser = User.objects.create_user(username, email, password1)
            myuser.save()
            messages.success(
                request, "your Accont has been Successfuly Created.")
            return render(request, "signin.html")
        else:
            messages.warning(request, "Password Not Match")
        return render(request, "signup.html")

    return render(request, "signup.html")


def signin(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(username=email, password=password)

        if user is not None:
            login(request, user)
            username = user.username
            messages.success(request, "Login successfly")
            return render(request, "index.html", {"username": username})
        else:
            messages.warning(request, "no user exists with these credentials")
            return render(request, "signin.html")
    return render(request, "signin.html")


def signout(request):
    logout(request)
    messages.success(request, "logged Out Successfuly!")
    return redirect("home")


def run_bot():
    obj = TradingAccount.objects.filter(username=User.username)
    if obj.run_trading == True:
        # CAll Trading funstion here(live-trading.py)

        result = function.delay()

        pass
    if obj.run_scanner == True:
        # Call Scanner Function here (scanner.py)
        pass


@login_required(login_url="signin")
def trading(request):
    if request.method == "POST":
        obj = TradingAccount.objects.filter(username=User.username)
        obj.api_key = request.POST.get("api_key")
        obj.run_trading = request.POST.get("run_trading")
        obj.run_scanner = request.POST.get("run_scanner")
        # obj.token_path = request.POST.get("token_path")
        obj.redirect_url = request.POST.get("redirect_url")
        obj.account_id = request.POST.get("account_id")
        obj.amount = request.POST.get("amount")
        obj.trading_strategy = request.POST.get("trading_strategy")
        obj.trading_symbols = request.POST.get("trading_symbols")
        obj.save()
        messages.warning(request, "Your Data is Store")
        return render(request, "home.html")
    else:
        return render(request, "home.html")


@login_required(login_url="signin")
def scanner(request):
    if request.method == "POST":
        obj = TradingAccount.objects.filter(username=User.username)
        obj.scanner_strategy = request.POST.get("strategies")
        obj.scanner_symbols = request.POST.get("symbols")
        obj.save()
        messages.warning(request, "Your Data is Store")
        return render(request, "home.html")
    else:
        return render(request, "home.html")
