import copy
import json
import os
import re
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from torch.nn import Parameter
import string

from root import ROOT_DIR

SAVE_DIR = os.path.join(ROOT_DIR, 'swiss-xlm-roberta-base')

# CLEAN FOLDER
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)

# SAVE ORIGINAL XLM-ROBERTA TOKENIZER IN THE NEW FORMAT
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
config = AutoConfig.from_pretrained('xlm-roberta-base')
tokenizer.save_pretrained(SAVE_DIR, legacy_format=False)
config.save_pretrained(SAVE_DIR)

# GET UNIQUE CHARS FROM THE DATA
get_chars_from_data = False
if get_chars_from_data:
    filenames = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    unique_chars = set()
    for filename in filenames:
        print(f"Processing {filename}")
        with open('data/' + filename) as file:
            for line in tqdm(file.readlines()):
                example = json.loads(line)
                unique_chars = unique_chars.union(set(example['text']))

    unique_chars_sorted = sorted(list(unique_chars))
    unique_chars_sorted_str = ''.join(unique_chars_sorted)
    print('unique characters: ', unique_chars_sorted_str)

# FIND USABLE TOKENS IN VOCABULARY
# Keep only numbers, latin script and other chars than make sense for DE, FR, IT.
latin_characters = 'a-zA-Z0-9ÊõüèëØÅçéá̊òąàôœå≅êûßÖíä§ÂÇñǗÈË"ã€$£øαÉÀĄïṣùâóÎÄúÓæöμÔìî'
data_characters = '!"#$%&\'()*+,-./0-9:;<=>?@A-Z\[\]^_`a-z|£§©«®°±·»ÀÂÄÅÇÈÉÊËÎÓÔÖ×ØÜßàáâãäåæçèéêëìíîïñòóôõö÷øùúûüĄąœʺ̧́̈̊αμṣ‐‘„•‰′⁄€→−≅≤●'
VOCAB = re.compile(f'[{data_characters}]+')
en_tokens = []
punkt_tokens = []
trivial_tokens = []
el_tokens = []
usable_tokens = []
not_usable = []
usable_ids = []
for original_token, id in tokenizer.vocab.items():
    token = original_token.translate(str.maketrans('', '', string.punctuation + '▁–⁄°€')).lower()
    if VOCAB.fullmatch(token) or len(token) == 0:
        usable_tokens.append(original_token)
        usable_ids.append(id)
    else:
        not_usable.append(original_token)

print(
    f'USABLE VOCABULARY: {len(usable_ids)}/{len(tokenizer.vocab)} ({(len(usable_ids) * 100) / len(tokenizer.vocab):.1f}%)')

# UPDATE TOKENIZER VOCABULARY
usable_tokens = set(usable_tokens)
with open(os.path.join(SAVE_DIR, 'tokenizer.json')) as file:
    tokenizer_data = json.load(file)
    tokenizer_data['model']['vocab'] = [token for token in tokenizer_data['model']['vocab'] if
                                        token[0] in usable_tokens]
    tokenizer_data['added_tokens'][-1]['id'] = len(tokenizer_data['model']['vocab'])

# SAVE TOKENIZER JSON
with open(os.path.join(SAVE_DIR, 'tokenizer.json'), 'w') as file:
    json.dump(tokenizer_data, file)

# HACK XLM-ROBERTA
print('HACK XLM-ROBERTA')

# LOAD XLM-ROBERTA
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
eu_model_pt = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')

# COLLECT USABLE (EMBEDDINGS + LM HEAD) WEIGHTS
usable_ids = set(usable_ids)
embeddings = copy.deepcopy(
    [embed for idx, embed in enumerate(eu_model_pt.roberta.embeddings.word_embeddings.weight.detach().numpy()) if
     idx in usable_ids])
lm_head_bias = copy.deepcopy(
    [embed for idx, embed in enumerate(eu_model_pt.lm_head.bias.detach().numpy()) if idx in usable_ids])
lm_head_decoder_bias = copy.deepcopy(
    [embed for idx, embed in enumerate(eu_model_pt.lm_head.decoder.bias.detach().numpy()) if idx in usable_ids])
lm_head_decoder_weight = copy.deepcopy(
    [embed for idx, embed in enumerate(eu_model_pt.lm_head.decoder.weight.detach().numpy()) if idx in usable_ids])

# REASSIGN USABLE WEIGHTS TO (EMBEDDINGS + LM HEAD) LAYERS
eu_model_pt.resize_token_embeddings(len(usable_ids))
eu_model_pt.roberta.embeddings.word_embeddings.weight = Parameter(torch.as_tensor(embeddings))
eu_model_pt.lm_head.bias = Parameter(torch.as_tensor(lm_head_bias))
eu_model_pt.lm_head.decoder.weight = Parameter(torch.as_tensor(lm_head_decoder_weight))
eu_model_pt.lm_head.decoder.bias = Parameter(torch.as_tensor(lm_head_decoder_bias))

# SAVE MODEL AND TOKENIZER
eu_model_pt.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR, legacy_format=False)

# TEST MODEL AS LANGUAGE MODEL
print('INFERENCE')
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
final_model = AutoModelForMaskedLM.from_pretrained(SAVE_DIR)


def test_sentence(model, sentence):
    model_output = model(input_ids=tokenizer(sentence, return_tensors='pt').input_ids)
    print(sentence)
    print(tokenizer.decode(torch.argmax(model_output.logits, dim=-1).numpy()[0]))
    print()


def test_sentences(model):
    with torch.no_grad():
        # English
        test_sentence(model, 'Her <mask> is hairy.')
        test_sentence(model, 'A <mask> sunny holiday.')
        test_sentence(model, 'He played <mask> guitar, while the other guy was playing piano.')
        test_sentence(model, 'Paris is the <mask> of France.')

        # German
        test_sentence(model, 'Paris ist die <mask> von Frankreich.')
        test_sentence(model, 'Hast du <mask> Rübe gekauft?')
        test_sentence(model, 'Wir <mask> März, bald ist Frühling.')

        # French
        test_sentence(model, 'Paris est la <mask> de la France.')
        test_sentence(model, 'Mon français s\'améliore de <mask> en jour.')
        test_sentence(model, 'J\'ai nettoyé <mask> mâche.')
        test_sentence(model, 'Nous <mask> à la plage.')

        # Italian
        test_sentence(model, 'Parigi è la <mask> della Francia.')
        test_sentence(model, 'Si può <mask> un po’ più forte, per favore?')
        test_sentence(model, 'Potrebbe <mask>, per favore?')

        # Greek
        test_sentence(model, 'Ο πρόεδρος του <mask>.')


print("Testing Swiss xlm-roberta-base")
test_sentences(final_model)

print("Testing general xlm-roberta-base")
test_sentences(eu_model_pt)
