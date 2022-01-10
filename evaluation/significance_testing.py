from collections import OrderedDict

from deepsig import aso, multi_aso

# Simulate scores
from arguments.data_arguments import DataAugmentationType, Jurisdiction
from arguments.model_arguments import TrainType
from evaluation.create_tables import isNativeBert
from utils.wandb_util import retrieve_results

project_name = "SwissJudgmentPredictionCrossLingualTransfer"
# Important overwrite_cache as soon as there are new results
original_df = retrieve_results(project_name, overwrite_cache=False)


def get_scores(original_df, filter):
    df = original_df[original_df.config.apply(filter)]
    scores = {"concatenated": []}
    for lang in ['de', 'fr', 'it']:
        key = f'test/{lang}/f1_macro'
        scores[lang] = df.summary.apply(lambda x: x[key] if key in x.keys() else None).dropna().tolist()
        scores["concatenated"].extend(scores[lang])
    return scores


scores = OrderedDict()

filter = lambda x: x['model_args']['train_languages'] in ['de', 'fr', 'it'] \
                   and x['data_args']['data_augmentation_type'] == DataAugmentationType.NO_AUGMENTATION \
                   and x['data_args']['train_sub_datasets'] == "None" \
                   and x['model_args']['train_type'] == TrainType.FINETUNE \
                   and isNativeBert(x['model_args']['model_name'])

scores['native_no_aug'] = get_scores(original_df, filter)

filter = lambda x: x['model_args']['train_languages'] in ['de', 'fr', 'it'] \
                   and x['data_args']['data_augmentation_type'] == DataAugmentationType.TRANSLATION \
                   and x['data_args']['train_sub_datasets'] == "None" \
                   and x['model_args']['train_type'] == TrainType.FINETUNE \
                   and isNativeBert(x['model_args']['model_name'])
scores['native_trans'] = get_scores(original_df, filter)

filter = lambda x: x['data_args']['jurisdiction'] == Jurisdiction.SWITZERLAND and \
                   x['model_args']['train_languages'] in ['de,fr,it'] \
                   and x['data_args']['data_augmentation_type'] == DataAugmentationType.TRANSLATION \
                   and x['data_args']['train_sub_datasets'] == "None" \
                   and x['model_args']['train_type'] == TrainType.FINETUNE \
                   and not isNativeBert(x['model_args']['model_name'])
scores['xlm_r_trans_ch'] = get_scores(original_df, filter)

filter = lambda x: x['data_args']['jurisdiction'] == Jurisdiction.BOTH
scores['xlm_r_trans_ch+in'] = get_scores(original_df, filter)

for subset in ["concatenated", "de", "fr", "it"]:
    subset_dict = {key: scores[subset] for key, scores in scores.items()}
    eps_min = multi_aso(subset_dict, confidence_level=0.05, return_df=True, num_jobs=16, seed=42)
    print(f"Computed Significances on subset {subset}")
    print()
    print(eps_min.to_string())
    eps_min.to_latex(f"significances_{subset}.tex")
