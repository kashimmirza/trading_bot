from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User


class TradingStrategy(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class Symbol(models.Model):
    symbol = models.CharField(max_length=10)

    def __str__(self):
        return self.symbol


class TradingAccount(models.Model):
    username = models.CharField(max_length=20)
    run_trading = models.BooleanField()
    run_scanner = models.BooleanField()
    api_key = models.CharField(max_length=255)
    token_path = models.CharField(max_length=255)
    redirect_url = models.URLField()
    account_id = models.CharField(max_length=255)
    trading_symbols = models.ManyToManyField(
        Symbol, related_name='trading_accounts')
    amount = models.IntegerField()
    trading_strategy = models.ForeignKey(
        TradingStrategy, on_delete=models.CASCADE)
    scanner_symbols = models.ManyToManyField(
        Symbol, related_name='scanner_accounts')
    scanner_strategy = models.ForeignKey(
        TradingStrategy, on_delete=models.CASCADE, related_name='scanner_strategy')

    def __str__(self):
        return self.account_id
