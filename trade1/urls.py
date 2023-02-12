from django.urls import path
from .views import create_trading_account, trading_account_list
import views

urlpatterns = [
    path('create/', create_trading_account, name='create_trading_account'),
    path('list/', trading_account_list, name='trading_account_list'),
    path("signup", views.signup, name="signup"),
    path("signin", views.signin, name="signin"),
    path("signout", views.signout, name="signout"),
    path("scanner", views.scanner, name="scanner"),
    # path("graph", views.graph, name="graph"),
    path("trading", views.trading, name="trading"),
    # path("account", views.account, name="account"),

]
