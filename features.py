
from functools import wraps

import pandas

import numpy as np

import pandas as pd

# Helpers

def px(pt, eta, phi):
    return np.where(pt >= 0, pt * np.cos(phi), -999)

def py(pt, eta, phi):
    return np.where(pt >= 0, pt * np.sin(phi), -999)

def pz(pt, eta, phi):
    return np.where(pt >= 0, pt * np.sinh(eta), -999)

def p_tot(pt, eta, phi):
    x = px(pt, eta, phi)
    y = py(pt, eta, phi)
    z = pz(pt, eta, phi)
    return np.sqrt(x*x + y*y + z*z)

def _calculate_momenta(df, prefix):
    pt = df[prefix+'pt']
    eta = df[prefix+'eta']
    phi = df[prefix+'phi']

    return pd.DataFrame({
        prefix+'px': px(pt, eta, phi),
        prefix+'py': py(pt, eta, phi),
        prefix+'pz': pz(pt, eta, phi),
        prefix+'p_tot': p_tot(pt, eta, phi)
    }, index=df.index).fillna(-999)


def total_maximum(xs):
    m = None
    for x in xs:
        if m is None:
            m = x
        else:
            m = np.maximum(m, x)
    return m


def total_minimum(xs):
    m = None
    for x in xs:
        if m is None:
            m = x
        else:
            m = np.minimum(m, x)
    return m


def require_jets(njets, default=-999):
    def feature_decorator(func):
        @wraps(func)
        def func_wrapper(df):
            return np.where(df.PRI_jet_num >= njets, func(df), default)
        return func_wrapper
    return feature_decorator


# The distance from the x=y line in jet eta0, eta1 space
def eta_plus(x, y):
    return np.sqrt(x*x/2 + 2*y*y - 2*x*y)


# Create momentum variables

def with_momentum_features(df):

    df = df.copy()

    lep = _calculate_momenta(df, 'PRI_lep_')
    jet_leading = _calculate_momenta(df, 'PRI_jet_leading_')
    jet_subleading = _calculate_momenta(df, 'PRI_jet_subleading_')
    tau = _calculate_momenta(df, 'PRI_tau_')

    for df_mom in [lep, jet_leading, jet_subleading, tau]:
        for col, srs in df_mom.iteritems():
           df[col] = srs

    return df

#
# Start creating features
#

#
# Eta Features
#

@require_jets(2)
def jet_eta_plus(df):
    x = df['PRI_jet_leading_eta']
    y = df['PRI_jet_subleading_eta']
    return np.where((x > -900) & (y > -900),
                    eta_plus(x, y),
                    -999)

# Do the same with the lepton and jet
def lep_tau_eta_plus(df):
    x = df['PRI_lep_eta']
    y = df['PRI_tau_eta']
    return eta_plus(x, y)

rapidity_features = [jet_eta_plus, lep_tau_eta_plus]

#
# Z Momentum Features
#

def lep_z_momentum(df):
    return df['PRI_lep_pz'] + df['PRI_tau_pz']

@require_jets(2)
def jet_z_momentum(df):
    return df['PRI_jet_leading_pz'] + df['PRI_jet_subleading_pz']

@require_jets(2)
def jet_lep_sum_z_momentum(df):
    return lep_z_momentum(df) + jet_z_momentum(df)

@require_jets(2)
def jet_lep_diff_z_momentum(df):
    return lep_z_momentum(df) - jet_z_momentum(df)

z_momentum_features = [lep_z_momentum, jet_z_momentum, jet_lep_sum_z_momentum, jet_lep_diff_z_momentum]

#
# Transverse Momenta Features
#

def max_jet_pt(df):
    return np.maximum(
        np.where(df['PRI_jet_leading_pt'] > 0, df['PRI_jet_leading_pt'], 0),
        np.where(df['PRI_jet_subleading_pt'] > 0, df['PRI_jet_subleading_pt'], 0))

def min_jet_pt(df):
    return np.minimum(
        np.where(df['PRI_jet_leading_pt'] > 0, df['PRI_jet_leading_pt'], 0),
        np.where(df['PRI_jet_subleading_pt'] > 0, df['PRI_jet_subleading_pt'], 0))

def max_lep_pt(df):
    return np.maximum(df['PRI_tau_pt'], df['PRI_lep_pt'])

def min_lep_pt(df):
    return np.minimum(df['PRI_tau_pt'], df['PRI_lep_pt'])

def max_pt(df):
    return np.maximum(max_jet_pt(df), max_lep_pt(df))

def min_pt(df):
    return np.minimum(min_jet_pt(df), min_lep_pt(df))

def sum_jet_pt(df):
    return np.where(df['PRI_jet_leading_pt'] > 0, df['PRI_jet_leading_pt'], 0) + np.where(df['PRI_jet_subleading_pt']>0, df['PRI_jet_subleading_pt'], 0)

def sum_lep_pt(df):
    return df['PRI_tau_pt'] + df['PRI_lep_pt']


transverse_momentum_features = [max_jet_pt, min_jet_pt, max_lep_pt, min_lep_pt,
                                max_pt, min_pt, sum_jet_pt, sum_lep_pt]

#
# Momentum Ratio Features
#

def frac_tau_pt(df):
    tau_pt = df['PRI_tau_pt']
    lep_pt = df['PRI_lep_pt']
    pt_sum = (tau_pt + lep_pt)
    return np.where(pt_sum > 0, tau_pt / pt_sum, -999)

def frac_lep_pt(df):
    tau_pt = df['PRI_tau_pt']
    lep_pt = df['PRI_lep_pt']
    pt_sum = (tau_pt + lep_pt)
    return np.where(pt_sum > 0, lep_pt / pt_sum, -999)

def frac_tau_p(df):
    tau_p = df['PRI_tau_p_tot']
    lep_p = df['PRI_lep_p_tot']
    p_sum = (tau_p + lep_p)
    return np.where(p_sum > 0, tau_p / p_sum, -999)

def frac_lep_p(df):
    tau_p = df['PRI_tau_p_tot']
    lep_p = df['PRI_lep_p_tot']
    p_sum = (tau_p + lep_p)
    return np.where(p_sum != 0, lep_p / p_sum, -999)


@require_jets(2)
def prijet_subjet_pt_ratio(df):
    return df['PRI_jet_leading_pt'] / (df['PRI_jet_leading_pt'] + df['PRI_jet_subleading_pt'])

@require_jets(2)
def subjet_prijet_pt_ratio(df):
    return df['PRI_jet_subleading_pt'] / (df['PRI_jet_leading_pt'] + df['PRI_jet_subleading_pt'])

momentum_ratio_features = [frac_tau_pt, frac_lep_pt, frac_tau_p, frac_lep_p,
                           prijet_subjet_pt_ratio, subjet_prijet_pt_ratio]



#
# Mass / Energy Features
#


def transverse_mass_lep(row):
    lep_pt = np.sqrt(row['PRI_lep_px']*row['PRI_lep_px'] + row['PRI_lep_py']*row['PRI_lep_py'])
    return np.sqrt(2*lep_pt*row['PRI_met'] * (1 - (np.cos(row['PRI_met_phi'] - row['PRI_lep_phi']))))

def transverse_mass_tau(row):
    tau_pt = np.sqrt(row['PRI_tau_px']*row['PRI_tau_px'] + row['PRI_tau_py']*row['PRI_tau_py'])
    return np.sqrt(2*tau_pt*row['PRI_met'] * (1 - np.cos(row['PRI_met_phi'] - row['PRI_tau_phi'])))

def transverse_mass_jet_leading(row):
    lep_pt = np.sqrt(row['PRI_jet_leading_px']*row['PRI_jet_leading_px'] + row['PRI_jet_leading_py']*row['PRI_jet_leading_py'])
    return np.sqrt(2*lep_pt*row['PRI_met'] * (1 - (np.cos(row['PRI_met_phi'] - row['PRI_jet_leading_phi']))))

def transverse_mass_jet_subleading(row):
    lep_pt = np.sqrt(row['PRI_jet_subleading_px']*row['PRI_jet_subleading_px'] + row['PRI_jet_subleading_py']*row['PRI_jet_subleading_py'])
    return np.sqrt(2*lep_pt*row['PRI_met'] * (1 - (np.cos(row['PRI_met_phi'] - row['PRI_jet_subleading_phi']))))


tau_mass = 1.7

def tau_fourenergy(row):
    tau_2 = row['PRI_tau_px']*row['PRI_tau_px'] + row['PRI_tau_py']*row['PRI_tau_py'] + row['PRI_tau_pz']*row['PRI_tau_pz']
    return np.sqrt(tau_mass*tau_mass + tau_2)

def lep_fourenergy(row):
    lep_2 = row['PRI_lep_px']*row['PRI_lep_px'] + row['PRI_lep_py']*row['PRI_lep_py'] + row['PRI_lep_pz']*row['PRI_lep_pz']
    return np.sqrt(tau_mass*tau_mass + lep_2)


def lep_mass(df):
    delta_cosh_eta = np.cosh(df['PRI_lep_eta'] - df['PRI_tau_eta'])
    delta_cos_phi = np.cos(df['PRI_lep_phi'] - df['PRI_tau_phi'])
    return 2*df['PRI_lep_pt']*df['PRI_tau_pt']*(delta_cosh_eta - delta_cos_phi)



mass_energy_features = [transverse_mass_lep, transverse_mass_tau, transverse_mass_jet_leading,
                        transverse_mass_jet_subleading, tau_fourenergy, lep_fourenergy, lep_mass]

#
# MET Features
#

def ht(df):
    return sum_jet_pt(df) + sum_lep_pt(df)

def ht_met(df):
    return ht(df) + df['PRI_met']

def tau_met_cos_phi(df):
    return np.cos(df['PRI_met_phi'] - df['PRI_tau_phi'])

def lep_met_cos_phi(df):
    return np.cos(df['PRI_met_phi'] - df['PRI_lep_phi'])

@require_jets(1)
def jet_leading_met_cos_phi(df):
    return np.cos(df['PRI_jet_leading_phi'] - df['PRI_lep_phi'])

@require_jets(2)
def jet_subleading_met_cos_phi(df):
    return np.cos(df['PRI_jet_subleading_phi'] - df['PRI_lep_phi'])

@require_jets(2)
def min_met_cos_phi(df):
    return total_minimum([
        tau_met_cos_phi(df),
        lep_met_cos_phi(df),
        jet_leading_met_cos_phi(df),
        jet_subleading_met_cos_phi(df)])

@require_jets(2)
def max_met_cos_phi(df):
    return total_maximum([
        tau_met_cos_phi(df),
        lep_met_cos_phi(df),
        jet_leading_met_cos_phi(df),
        jet_subleading_met_cos_phi(df)])

def met_sig(df):
    return np.where(df['PRI_met_sumet'] > 0,
                    df['PRI_met'] / np.sqrt(df['PRI_met_sumet']),
                    -999)

def sumet_sum_pt_ratio(df):
    return np.where(df['DER_sum_pt'] != 0,
                    df['PRI_met_sumet'] / df['DER_sum_pt'],
                    -999)

def met_pt_total_ratio(df):
    return np.where(df['DER_pt_tot']==0, -999, df['PRI_met'] / df['DER_pt_tot'])

met_features = [ht, ht_met, tau_met_cos_phi, lep_met_cos_phi, jet_leading_met_cos_phi, jet_subleading_met_cos_phi,
                min_met_cos_phi, max_met_cos_phi, met_sig, sumet_sum_pt_ratio, met_pt_total_ratio]

#
# Jet Features
#

@require_jets(2)
def jet_delta_cos_phi(df):
    return np.cos(df['PRI_jet_leading_phi'] - df['PRI_jet_subleading_phi'])

def prijet_tau_delta_cos_phi(row):
    return np.cos(df['PRI_jet_leading_phi'] - df['PRI_tau_phi'])

def prijet_lep_delta_cos_phi(df):
    return np.cos(df['PRI_jet_leading_phi'] - df['PRI_lep_phi'])

def subjet_lep_delta_cos_phi(df):
    return np.cos(df['PRI_jet_subleading_phi'] - df['PRI_tau_phi'])

def subjet_lep_delta_cos_phi(df):
    return np.cos(df['PRI_jet_subleading_phi'] - df['PRI_lep_phi'])

def lep_tau_delta_cos_phi(df):
    return np.cos(df['PRI_lep_phi'] - df['PRI_tau_phi'])



jet_features = [jet_delta_cos_phi, prijet_tau_delta_cos_phi, prijet_lep_delta_cos_phi, subjet_lep_delta_cos_phi,
                subjet_lep_delta_cos_phi, lep_tau_delta_cos_phi]



# Adding features to a DF

def with_added_features(df):

    df = df.copy()

    print "Adding momentum features"
    df = with_momentum_features(df)

    new_features = []
    new_features.extend(rapidity_features)
    new_features.extend(z_momentum_features)
    new_features.extend(transverse_momentum_features)
    new_features.extend(met_features)
    new_features.extend(momentum_ratio_features)
    new_features.extend(mass_energy_features)

    for f in new_features:
        name = f.__name__
        print "Calculating {}".format(name)
        df[name] = pd.Series(f(df), index=df.index).replace([np.inf, -np.inf], np.nan).fillna(-999)

    return df


