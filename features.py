
import pandas

import numpy as np

import math
import math
from math import sin, cos, sinh

from bamboo.data import map_functions
from bamboo.bamboo import threading



def jet_partition(row):
    jet_num = row['PRI_jet_num']
    if jet_num==0:
        return 'zero_jet'
    elif jet_num==1:
        return 'one_jet'
    else:
        return 'multi_jet'

# Momentum

def px(pt, eta, phi):
    return pt * np.cos(phi)

def py(pt, eta, phi):
    return pt * np.sin(phi)

def pz(pt, eta, phi):
    return pt * np.sinh(eta)


def p_tot(pt, eta, phi):
    x = px(pt, eta, phi)
    y = py(pt, eta, phi)
    z = pz(pt, eta, phi)
    return np.sqrt(x*x + y*y + z*z)


def _calculate_momenta(df, prefix):
    pt = df[prefix+'pt']
    eta = df[prefix+'eta']
    phi = df[prefix+'phi']

    return pandas.DataFrame({prefix+'px' : px(pt, eta, phi),
                      prefix+'py' : py(pt, eta, phi),
                      prefix+'pz' : pz(pt, eta, phi),
                      prefix+'p_tot' : p_tot(pt, eta, phi)})


def get_momentum_features(df):
    lep = _calculate_momenta(df, 'PRI_lep_')
    jet_leading = _calculate_momenta(df, 'PRI_jet_leading_')
    jet_subleading = _calculate_momenta(df, 'PRI_jet_subleading_')
    tau = _calculate_momenta(df, 'PRI_tau_')

    return lep.join(tau).join(jet_leading).join(jet_subleading)


def with_momentum_features(df):
    return df.join(get_momentum_features(df)).replace([np.inf, -np.inf], np.nan).fillna(0)



# Eta Features

# The distance from the x=y line in jet eta0, eta1 space
def eta_plus(x, y):
    return np.sqrt(x*x/2 + 2*y*y - 2*x*y)

def jet_eta_plus(row):
    x = row['PRI_jet_leading_eta']
    y = row['PRI_jet_subleading_eta']
    return eta_plus(x, y)

# Do the same with the lepton and jet
def lep_tau_eta_plus(row):
    x = row['PRI_lep_eta']
    y = row['PRI_tau_eta']
    return eta_plus(x, y)

rapidity_features = [jet_eta_plus, lep_tau_eta_plus]


# Z Momentum Features

def lep_z_momentum(row):
    return row['PRI_lep_pz'] + row['PRI_tau_pz']

def jet_z_momentum(row):
    return row['PRI_jet_leading_pz'] + row['PRI_jet_subleading_pz']

def jet_lep_sum_z_momentum(row):
    return lep_z_momentum(row) + jet_z_momentum(row)

def jet_lep_diff_z_momentum(row):
    return lep_z_momentum(row) - jet_z_momentum(row)

z_momentum_features = [lep_z_momentum, jet_z_momentum, jet_lep_sum_z_momentum, jet_lep_diff_z_momentum]


# Transverse Momenta Features

def max_jet_pt(row):
    return max(row['PRI_jet_leading_pt'], row['PRI_jet_subleading_pt'])  

def min_jet_pt(row):
    return min(row['PRI_jet_leading_pt'], row['PRI_jet_subleading_pt'])  

def max_lep_pt(row):
    return max(row['PRI_tau_pt'], row['PRI_lep_pt'])  

def min_lep_pt(row):
    return min(row['PRI_tau_pt'], row['PRI_lep_pt'])  

def max_pt(row):
    return max(max_jet_pt(row), max_lep_pt(row))

def min_pt(row):
    return min(min_jet_pt(row), min_lep_pt(row))

def sum_jet_pt(row):
    return row['PRI_jet_leading_pt'] + row['PRI_jet_subleading_pt']

def sum_lep_pt(row):
    return row['PRI_tau_pt'] + row['PRI_lep_pt']


transverse_momentum_features = [max_jet_pt, min_jet_pt, max_lep_pt, min_lep_pt,
                                max_pt, min_pt, sum_jet_pt, sum_lep_pt]


# Momentum Ratio Features

def frac_tau_pt(row):
    tau_pt = row['PRI_tau_pt']
    lep_pt = row['PRI_lep_pt']
    return tau_pt / (tau_pt + lep_pt)

def frac_lep_pt(row):
    tau_pt = row['PRI_tau_pt']
    lep_pt = row['PRI_lep_pt']
    return lep_pt / (tau_pt + lep_pt)

def frac_tau_p(row):
    tau_p = row['PRI_tau_p_tot']
    lep_p = row['PRI_lep_p_tot']
    return tau_p / (tau_p + lep_p)

def frac_lep_p(row):
    tau_p = row['PRI_tau_p_tot']
    lep_p = row['PRI_lep_p_tot']
    return lep_p / (tau_p + lep_p)

momentum_ratio_features = [frac_tau_pt, frac_lep_pt, frac_tau_p, frac_lep_p]


# MET Features

def ht(row):
    return sum_jet_pt(row) + sum_lep_pt(row)

def ht_met(row):
    return ht(row) + row['PRI_met']

def tau_met_cos_phi(row):
    return math.cos(row['PRI_met_phi'] - row['PRI_tau_phi'])

def lep_met_cos_phi(row):
    return math.cos(row['PRI_met_phi'] - row['PRI_lep_phi'])

def jet_leading_met_cos_phi(row):
    return math.cos(row['PRI_jet_leading_phi'] - row['PRI_lep_phi'])

def jet_subleading_met_cos_phi(row):
    return math.cos(row['PRI_jet_subleading_phi'] - row['PRI_lep_phi'])

def min_met_cos_phi(row):
    return min(tau_met_cos_phi(row), lep_met_cos_phi(row),
               jet_leading_met_cos_phi(row), jet_subleading_met_cos_phi(row))

def max_met_cos_phi(row):
    return max(tau_met_cos_phi(row), lep_met_cos_phi(row),
               jet_leading_met_cos_phi(row), jet_subleading_met_cos_phi(row))

def met_sig(row):
    return row['PRI_met'] / np.sqrt(row['PRI_met_sumet'])

def sumet_sum_pt_ratio(row):
    return row['PRI_met_sumet'] / row['DER_sum_pt']

def met_pt_total_ratio(row):
    if (row['DER_pt_tot']==0):
        return 0.0
    return row['PRI_met'] / row['DER_pt_tot']

met_features = [ht, ht_met, tau_met_cos_phi, lep_met_cos_phi, jet_leading_met_cos_phi, jet_subleading_met_cos_phi,
                min_met_cos_phi, max_met_cos_phi, met_sig, sumet_sum_pt_ratio, met_pt_total_ratio]


# Jet Features

def jet_delta_cos_phi(row):
    return cos(row['PRI_jet_leading_phi'] - row['PRI_jet_subleading_phi'])

jet_features = [jet_delta_cos_phi]


# Adding features to a DF



def add_features(df):

    new_features = []
    new_features.extend(rapidity_features)
    new_features.extend(z_momentum_features)
    new_features.extend(transverse_momentum_features)
    new_features.extend(met_features)
    new_features.extend(momentum_ratio_features)


    def with_new_features(df):
        return df.join(map_functions(df, new_features))

    df_all_features = threading(df,
                                with_momentum_features,
                                with_new_features)

    return df_all_features



def main():
    pass



if __name__=='__main__':
    main()

