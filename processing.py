import sys
import os
import re
import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit
from copy import deepcopy

log = logging.getLogger()
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def load_and_concat_dfs(dpath, label):
    name_re = re.compile('{}.(\d+)-(\d+).pkl'.format(label))
    fnames = [fname for fname in os.listdir(dpath) if name_re.search(fname)]
    fnames.sort(key = lambda s: int(name_re.search(s).group(1)))
    
    m = name_re.search(fnames[0])
    start1, end1 = map(int, (m.group(1), m.group(2)))
    m = name_re.search(fnames[-1])
    start2, end2 = map(int, (m.group(1), m.group(2)))
    log.info('{}: {:,d}-{:,d}  ({:,d} files)'.format(label, start1, end2, len(fnames)))
    last_end = end2
    
    gap_total = 0
    for fname1, fname2 in zip(fnames, fnames[1:]):
        m = name_re.search(fname1)
        start1, end1 = map(int, (m.group(1), m.group(2)))
        m = name_re.search(fname2)
        start2, end2 = map(int, (m.group(1), m.group(2)))
        gap = start2 - end1 - 1
        if not gap == 0:
            log.info('Missing file between {} {} ({:,d} missing)'.format(fname1, fname2, gap))
            gap_total += gap
    log.info('Total missing: {:,d}'.format(gap_total))
            
    dfs = [pd.read_pickle(os.path.join(dpath, fname)) for fname in fnames]
    return pd.concat(dfs)


def calculate_y0s_and_vks(
    limit_of_counts,
    run_label,
    sample_names,
    cut_samples,
    times,
    oligo_container,
    read_names_given_sample,
    bootstrap=False,
    verbose=False
    
):
    target_tab = pd.read_pickle('{}_target_tab.pkl'.format(run_label))
    target_tab = target_tab[sample_names]

    nontarget_tab = pd.read_pickle('{}_nontarget_tab.pkl'.format(run_label))
    nontarget_tab = nontarget_tab[sample_names]

    raw_norm_factors = list(np.loadtxt('{}_norm_factors.txt'.format(run_label)))
    
    target_tab_copy = deepcopy(target_tab)
    y = [len(read_names_given_sample[sample]) for sample in sample_names]
    target_tab_totals = pd.DataFrame({sample: yy
                                      for sample, yy in zip(sample_names, y)}, index=['rest'])
    target_tab_sum = target_tab_copy.sum()
    target_tab_rest = target_tab_totals - target_tab_sum
    target_tab_copy = target_tab_copy.append(target_tab_rest)
    target_tab_multinom = target_tab_copy.div(list(target_tab_totals.iloc[0]), axis='columns')
    target_tab_multinom = target_tab_multinom[sample_names]

    assert all(target_tab.index == target_tab_multinom.index[:-1])
    cut_success = False
    while not cut_success:
        if bootstrap:
            if verbose: 
                print 'Calculating bootstrap read counts...'

            for sample in sample_names:
                target_tab[sample] = np.random.multinomial(target_tab_totals[sample], target_tab_multinom[sample])[:-1]

        if verbose: 
            print 'Normalizing...'

        norm_target_tab = target_tab.div(raw_norm_factors, axis='columns')
        norm_target_tab = norm_target_tab[sample_names]

        norm_perfect_target_tab = norm_target_tab.loc[[oligo.sequence for oligo in oligo_container.perfect_target_oligos]]
        #print norm_perfect_target_tab
        
        norm_nontarget_tab = nontarget_tab.div(raw_norm_factors, axis='columns')
        norm_nontarget_tab = norm_nontarget_tab[sample_names]

        nontarget_medians = list(norm_nontarget_tab.median())
        ref_norm_factors = np.array(nontarget_medians) / nontarget_medians[0]

        med_norm_target_tab = norm_target_tab.div(ref_norm_factors, axis='columns')
        med_norm_target_tab = med_norm_target_tab[sample_names]

        med_norm_perfect_target_tab = norm_perfect_target_tab.div(ref_norm_factors, axis='columns')
        med_norm_perfect_target_tab = med_norm_perfect_target_tab[sample_names]

        med_norm_nontarget_tab = norm_nontarget_tab.div(ref_norm_factors, axis='columns')
        med_norm_nontarget_tab = med_norm_nontarget_tab[sample_names]

        cut_target_tab = target_tab[cut_samples]
        med_norm_cut_target_tab = med_norm_target_tab[cut_samples]

        if verbose:
            print 'Calculating exp floor...'
        df = med_norm_perfect_target_tab.loc[:, cut_samples]
        perfect_final_fracs = []
        for index, row in df.iterrows():
            if y[0] >= limit_of_counts: 
                y = np.array(row)
                y /= y[0]
                perfect_final_fracs.append(y[-1])
        exp_floor = np.median(perfect_final_fracs)

        def exponential(x, y0, vk):
            return y0 * ((1 - exp_floor) * np.exp(-vk * x) + exp_floor)

        def find_vk_0(seq_read_counts):
            halfway = 0.5 * seq_read_counts[0]
            for tA, tB, countA, countB in zip(times, times[1:], seq_read_counts, seq_read_counts[1:]):
                if countB <= halfway:
                    assert countA > halfway, counts
                    t_half = tA + (tB - tA) * (halfway - countA)/(countB - countA)
                    return np.log(2)/t_half
            return np.log(2)/times[-1]

        def curve_fit_cut_data(seq_read_counts):
            seq_read_counts = np.array(seq_read_counts) + 0.1
            y0_0 = seq_read_counts[0]
            vk_0 = find_vk_0(seq_read_counts)
            return curve_fit(exponential, times, seq_read_counts, p0=[y0_0, vk_0],
                             bounds=((0, 0), (np.inf, np.inf)), method='trf', max_nfev=10000)

        if verbose:
            print 'Fitting...'

        vks, y0s = [], []
        covs = []
        cut_success = True
        for i in range(len(target_tab)):
            try:
                test_counts = np.array(list(med_norm_cut_target_tab.iloc[i]))
                popt, pcov = curve_fit_cut_data(test_counts)
                y0s.append(popt[0])    
                vks.append(popt[1])
                covs.append(pcov)
            except:
                cut_success = False
                sys.stdout.write('^')
                break
        if not cut_success:
            continue

        vk_y0_given_oligo_seq = {}
        kkm_given_target = {}
        for oligo_seq, y0, vk in zip(med_norm_cut_target_tab.index, y0s, vks):
            vk_y0_given_oligo_seq[oligo_seq] = (vk, y0)

        if verbose:
            print 'Done'
        return (
            target_tab,
            norm_target_tab,
            norm_perfect_target_tab,
            med_norm_target_tab,
            med_norm_perfect_target_tab,
            med_norm_cut_target_tab,
            nontarget_tab,
            norm_nontarget_tab,
            med_norm_nontarget_tab,
            raw_norm_factors,
            ref_norm_factors,
            perfect_final_fracs,
            exp_floor,
            exponential,
            vk_y0_given_oligo_seq,
        )
