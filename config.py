chromedriver_path=r'chromedriver.exe'

acc1_api_key = 'CX74SYMEDUUCIDYTNBBD5HUXVLLNO1UC@AMER.OAUTHAP'
acc1_token_path = '/home/software34/trading-bot/token'
acc1_redirect_uri = 'http://software34.pythonanywhere.com/'

acc_id = 253609860
symbols = ["AGYS", "TMDX"]
#symbols = ["NATH", "POWL", "RETA", "TITN", "VTYX", "AEHR", "FSS", "GFF"]"CAH", "ELF", "ESE", "SMCI",
amount = 50
strategy = "model_t-dqn_GOOG_10"
#strategy = ["model_dqn_GOOG_50" , "model_t-dqn_GOOG_10" ,  "model_double-dqn_GOOG_50"]



"""

  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]

"""