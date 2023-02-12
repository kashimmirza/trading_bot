from django import forms
from .models import TradingAccount, TradingStrategy, Symbol


class TradingAccountForm(forms.ModelForm):
    class Meta:
        model = TradingAccount
        fields = [
            'api_key',
            'token_path',
            'redirect_url',
            'account_id',
            'trading_symbols',
            'amount',
            'trading_strategy',
            'scanner_symbols',
            'scanner_strategy'
        ]
        widgets = {
            'trading_symbols': forms.CheckboxSelectMultiple,
            'scanner_symbols': forms.CheckboxSelectMultiple,
        }
