import numpy as np

from Code.PT_SSL_U.martingale import wsr_p_value

def compute_Pareto_frontier(costs):

    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_efficient[i]:
            continue
        dominated = (np.all(costs[is_efficient] >= costs[i], axis=1) & np.any(costs[is_efficient] > costs[i], axis=1))
        is_efficient[np.where(is_efficient)[0][dominated]] = False

    return is_efficient

def compute_p_values_wsr(losses, combinations, alphas, delta, Kmax):
    p_values = []

    for combi_index in combinations:
        batch_MD = losses.loc[losses['config_index'] == combi_index, 'Loss_MD']
        batch_MC = losses.loc[losses['config_index'] == combi_index, 'Loss_MC']

        batch_MD = batch_MD.to_numpy()
        batch_MC = batch_MC.to_numpy()

        batch_MD_per = np.random.permutation(batch_MD)
        batch_MC_per = np.random.permutation(batch_MC)

        pval_md = wsr_p_value(batch_MD_per, alphas[0], delta, A=0, B=Kmax-1, scale = 1)
        pval_mc = wsr_p_value(batch_MC_per, alphas[1], delta, A=0, B=Kmax, scale = 1)
        pval_md_rev = wsr_p_value(batch_MD_per[::-1], alphas[0], delta, A=0, B=Kmax - 1, scale=1)
        pval_mc_rev = wsr_p_value(batch_MC_per[::-1], alphas[1], delta, A=0, B=Kmax, scale=1)

        p_values.append(max(pval_md, pval_mc, pval_md_rev, pval_mc_rev))

    p_values = np.array(p_values)

    return p_values

def compute_risks(data, Kmax):
    data["Loss_Area"] = (data["Loss_Area"] / data["Estimated_speakers"].replace(0, np.nan)).fillna(0)

    agg_dict = {
        'Loss_MD': 'mean',
        'Loss_MC': 'mean',
        'Loss_FA': 'mean',
        'Loss_Area': 'mean',
        'Threshold_MD': 'first'
    }

    existing_cols = set(data.columns)
    for i in range(1, Kmax + 1):
        col = f'Threshold_MC_{i}'
        if col in existing_cols:
            agg_dict[col] = 'first'

    results = data.groupby('config_index').agg(agg_dict).reset_index().rename(columns={
        'Loss_MD': 'Risk_MD',
        'Loss_MC': 'Risk_MC',
        'Loss_FA': 'Risk_FA',
        'Loss_Area': 'Risk_Area'
    })

    config_index = results.pop('config_index')
    results['config_index'] = config_index

    results = results.round(4)

    return results