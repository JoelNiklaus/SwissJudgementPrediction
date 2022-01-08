"""
Compute legal area distributions across origin region subsets per data splits,
normalize to probability distribution,
compute wasserstein distance between train sets with test sets, another table
"""
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

from arguments.data_arguments import OriginRegion, LegalArea
from root import DATA_DIR


distributions = {}

for split in ["train", "test"]:  # we are not interested in validation
    distributions[split] = {}
    files = [DATA_DIR / lang / f"{split}.csv" for lang in ["de", "fr", "it"]]
    df = pd.concat(map(pd.read_csv, files))  # read all languages at the same time
    df.legal_area = df.legal_area.replace(["insurance_law", "other"], np.nan)
    df = df.dropna(subset=['origin_region', 'legal_area'])

    for region in OriginRegion:
        region_df = df[df.origin_region.str.contains(region)]
        distribution = region_df.legal_area.value_counts(normalize=True).to_dict()
        for legal_area in LegalArea:
            if legal_area not in distribution.keys():
                distribution[legal_area.value] = 0  # make sure, that all distributions have all legal areas
        distributions[split][region.value] = distribution

pprint(distributions)


def distribution_to_standardized_list(distribution: dict):
    """return a standardized list which always has the same order"""
    return [distribution[legal_area] for legal_area in LegalArea]


result = {}
for region_train, train_dist in distributions["train"].items():
    result[region_train] = {}
    for region_test, test_dist in distributions["test"].items():
        distance = wasserstein_distance(distribution_to_standardized_list(train_dist),
                                        distribution_to_standardized_list(test_dist))
        result[region_train][region_test] = round(distance, 3)

# train set in the columns, test set in the rows
result_df = pd.DataFrame.from_dict(result)
result_df.to_latex("legal_area_distribution_distances.tex")