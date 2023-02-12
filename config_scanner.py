chromedriver_path=r'chromedriver.exe'

acc1_api_key = 'CX74SYMEDUUCIDYTNBBD5HUXVLLNO1UC@AMER.OAUTHAP'
acc1_token_path = '/home/software34/trading-bot/token'
acc1_redirect_uri = 'http://software34.pythonanywhere.com/'

acc_id = 253609860
symbols = ["AGYS" , "PCG" , "TSLA","ADMA","AGI","AGYS","ALDX","AM","AMLX","AMPY","ARDX","ARL","ASC","ASRT","ASUR","ATI","ATXS","AVEO","AVTE","BELFB","CAAP","CAAS","CECO","CEG","CEIX","CEPU","CMPX","CNCE","COLL","CPI","CPRX","CRT","CVRX","DCPH","EDAP","EDN","EDU","ELF","FENC","FTI","GERN","GNE","HDSN","HGBL","HNRG","IBEX","IMCR","IMRA","IMVT","INSW","KDNY","LYTS","MLTX","MOD","MVO","MYOV","NINE","NVCN","PAM","PBF","PBYI","PDS","PFMT","PMTS","PWSC","PXS","RYTM","SBR","SGFY","SGML","SLB","SMCI","SMHI","STLD","STNG","SURG","SWIR","TAL","TCOM","TDW","TEDU","TGS","TH","TKC","TMDX","TNK","TNP","TRMD","TUSK","UFPT","ULH","VECT","VIPS","VIST","VOC","VRNA","VTRU","VVNT","WDH","WFRD","WLFC","YPF","ZYXI"]
amount = 50
strategy = ["model_dqn_GOOG_50" , "model_t-dqn_GOOG_10" ,  "model_double-dqn_GOOG_50"]



"""

  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]

"""