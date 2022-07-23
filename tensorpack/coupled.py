import xarray as xr
import numpy as np
import pandas as pd

def genSample():
    return xr.Dataset(
        data_vars=dict(
            flop=(["month", "time", "people", "state"], np.random.rand(8, 7, 6, 5)),
            turn=(["month", "time", "state"], np.random.rand(8, 7, 5)),
            river=(["month", "suit"], np.random.rand(8, 4)),
        ),
        coords=dict(
            month=["January", "February", "March", "April", "May", "June", "July", "August"],
            time=pd.date_range("2014-09-06", periods=7),
            people=["Liam", "Olivia", "Noah", "Emma", "Benjamin", "Charlotte"],
            state=["Ohio", "Tennessee", "Utah", "Virginia", "Wyoming"],
            suit=["Spade", "Heart", "Club", "Diamond"]
        ),
    )





def unfold(x: xr.Dataset, mode: str):
    """ Generate the flatten array along the mode axis """
    arrs = []  # save flattened arrays
    for varname, da in x.data_vars.items():
        if mode in da.coords:
            # unfold
            pass
        print(da.attrs)

    pass


def initDecomp(x: xr.Dataset, n_comp: int, method = "svd"):
    """ Setup the initial object; use SVD to initialize factors """
    pass



def CTF():



    pass
