import json
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as torch_utils
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def load_data(train_bound, val_bound, features, labels, batch_size=32):
    """
    :param train_bound: defines where we split data
    :param features: list matrix of features
    :param batch_size: nb of features too feed the net at each time point
    :return: nicely shuffled DataLoader objects
    """

    X = torch.tensor(features, requires_grad=True, dtype=torch.float)
    Y = torch.tensor(labels, requires_grad=False, dtype=torch.float)

    trainset = torch_utils.TensorDataset(X[:train_bound], Y[:train_bound])
    tr_load = torch_utils.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)

    val_set = torch_utils.TensorDataset(X[train_bound:val_bound], Y[train_bound:val_bound])
    val_load = torch_utils.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    test_set = torch_utils.TensorDataset(X[val_bound:], Y[val_bound:])
    te_load = torch_utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return tr_load, val_load, te_load


class RNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        self.hidden_size = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, bias=False, dropout=0.3)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, days, h0):

        days = days.unsqueeze(0)

        h_out, hidden = self.gru(days, h0)

        h_out = h_out.squeeze(0)

        out = self.fc(h_out)

        return out, hidden

    def initHidden(self, batch_size):

        return torch.zeros(1, batch_size, self.hidden_size)


def train(trainloader, val_loader, in_size, N, learning_rate):
    """
    :param trainloader: train dataLoader object
    :param in_size: Input dimension
    :param N: Batch Size
    :return: Saves params, outputs hidden weights for fctn
    """

    epochs = 1000

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpts["optim_state_dict"])

    loss_fn = nn.MSELoss()

    hn = net.initHidden(N)

    for epoch in range(epochs):

        average_loss = 0
        val_loss = 0

        for days, true_stats in trainloader:

            days = days.view(N, in_size)

            preds, hn = net(days, hn.detach())
            true_stats = true_stats.view(N, 70)
            loss = loss_fn(preds, true_stats)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            average_loss += loss.item()

        for days, truth in val_loader:

            days = days.view(N, in_size)

            preds, hn = net(days, hn)

            loss = loss_fn(preds, truth.reshape(32,70))

            val_loss += loss.item()

        if (epoch % 2) == 0:

            print("(epoch, train_loss, val_loss) = ({0}, {1}, {2})".format(epoch, average_loss/N, \
                                                                       val_loss/N))

    torch.save({'model_state_dict':net.state_dict(), 'optim_state_dict':optimizer.state_dict(), 'hn': hn}, \
               "model.pt")

    return hn


def normalize_data(og_df):
    """
    MinMax Scaling for input data
    """
    og1_df = og_df.copy
    df = og_df
    min_max_scaler = MinMaxScaler()
    mins = []
    scales = []
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))

    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
    m2 = min_max_scaler.min_
    s2 = min_max_scaler.scale_
    scales.append(s2)
    mins.append(m2)
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Close'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Price_x'] = min_max_scaler.fit_transform(df.Price_x.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Price_y'] = min_max_scaler.fit_transform(df.Price_y.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Price_x1'] = min_max_scaler.fit_transform(df.Price_x1.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Price_y1'] = min_max_scaler.fit_transform(df.Price_y1.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)
    df['Price'] = min_max_scaler.fit_transform(df.Price.values.reshape(-1,1))
    mins.append(min_max_scaler.data_min_)
    scales.append(min_max_scaler.scale_)


    return df


def test(testloader, hn, delta, batch):

    loss_fn = nn.MSELoss()

    final_preds = []
    truths = []

    total_loss = 0
    nb_samples = 0
    for days, truth in testloader:

        days = days.view(batch, delta*10)
        preds, _ = net(days, hn)
        truth = truth.view(batch, 70)
        loss = loss_fn(preds, truth)
        truths += truth.tolist()
        final_preds += preds.tolist()


        nb_samples += 1

        total_loss += loss.item()

    return final_preds[-1]


def unnormalize(min_max_scaler, n_v):
    """
    df: originial df
    n_v: normalized results
    """
    #df = normalize_data(df)

    #min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    scales = []
    mins = []

    n_v['Open'] = min_max_scaler.inverse_transform(n_v.Open.values.reshape(-1,1))
    m1 = min_max_scaler.min_
    s1 = min_max_scaler.scale_
    scales.append(s1)
    mins.append((m1))

    n_v['High'] = min_max_scaler.inverse_transform(n_v.High.values.reshape(-1, 1))
    m2 = min_max_scaler.min_
    s2 = min_max_scaler.scale_
    scales.append(s2)
    mins.append(m2)
    n_v['Low'] = min_max_scaler.inverse_transform(n_v.Low.values.reshape(-1, 1))
    n_v['Close'] = min_max_scaler.inverse_transform(n_v.Close.values.reshape(-1, 1))
    n_v['Price_x'] = min_max_scaler.inverse_transform(n_v.Price_x.values.reshape(-1, 1))
    n_v['Price_y'] = min_max_scaler.inverse_transform(n_v.Price_y.values.reshape(-1, 1))
    n_v['Price_x1'] = min_max_scaler.inverse_transform(n_v.Price_x1.values.reshape(-1, 1))
    n_v['Price_y1'] = min_max_scaler.inverse_transform(n_v.Price_y1.values.reshape(-1, 1))
    n_v['Price'] = min_max_scaler.inverse_transform(n_v.Price.values.reshape(-1, 1))

    print(mins)
    return n_v


def readbigData(delta, tick=None):
    """
    :param delta: defines how far back in time you're looking at
    :return: nice features & labels
    """
    features = []
    labels = []
    good_ones = ["VSLR", "RUN", "CSIQ","PEGI","FSLR","SPWR","JKS","ENPH","SEDG", "INE", "TSLA","AQN","DQ","AZRE","ASTI","FP", "YGEHY","SUNW","RGSE","EVSI"]
    if tick != None:
        good_ones = [tick]
    minmaxes = []
    for ticker in good_ones:

        # for some companies I was not able to get the prices
        try:
            df = pd.read_csv("./data/prices/"+ticker+".csv")
            df.drop('Date',1, inplace=True)
            df.drop(df.tail(1).index, inplace=True)
            mm = MinMaxScaler()
            mm.fit(df)

            normed = normalize_data(df)
            mat = normed.as_matrix()

            # TWO DAY PREDICTIONS (passing delta is 50, but only 49 days in feature matrix)
            for i in range(len(df.High) - delta - 7):
                features.append(mat[i:i+delta-1])
                labels.append(mat[i+delta: i+delta+7])

        except FileNotFoundError:
            continue
    return features, labels, df, mm

# DELTA: HOW FAR BACK YOU HAVE STOCK INFO FOR
delta = 49

in_dim = delta*10 # EACH SAMPLE HAS 9 FEATURES
hidden_dim = 49
out_dim = 10*7
lr = 1e-4
batch = 32
net = RNNClassifier(in_dim, hidden_dim, out_dim)

# load weights from previous session
checkpts = torch.load("model.pt")
net.load_state_dict(checkpts['model_state_dict'])
hidden_weight = checkpts['hn']

def final():
    good_ones = ["VSLR", "RUN", "CSIQ","PEGI","FSLR","SPWR","JKS","ENPH","SEDG", "TSLA","AQN","DQ","AZRE","ASTI", "YGEHY","SUNW","RGSE","EVSI"]

    for TICKER in good_ones:
        print(TICKER)
        feats, labels, df, mm = readbigData(50, tick=TICKER)

        # Days = number of samples that we have
        days = len(feats)
        trainloader, valloader, testloader = load_data(int(days * 0.1), int(days * 0.2), features=feats, labels=labels)



        # hidden_weight = train(trainloader, valloader, in_dim, batch, learning_rate=lr)

        X = test(testloader, hidden_weight, delta, batch)

        final_pred = pd.DataFrame(X)
        X = np.array(X).reshape(7,10)

        for i in range(7):
            df = df.append({'High':X[i][0], 'Low':X[i][1], 'Open':X[i][2], 'Close':X[i][3], 'Price_x':X[i][4], \
                            'Price_y':X[i][5], 'Price_x1':X[i][6], 'Price_y1':X[i][7], 'Price':X[i][8], 'news polarity':0}, ignore_index=True)

        df = mm.inverse_transform(df)

        print(df[:-7])
        d = pd.DataFrame(df)
        d.to_csv(path_or_buf="{}_preds.csv".format(TICKER))

final()







# want to append the last 70 to csv file
#print(out_dict['predictions'][-1])



