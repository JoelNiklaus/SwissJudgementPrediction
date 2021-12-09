from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
print('Tokenizer vocab: ', len(tokenizer))
# or petitioner and respondent
num_added_toks = tokenizer.add_tokens(['<plaintiff>', '<defendant>'], special_tokens=True)
print('We have added', num_added_toks, 'tokens')
print('Tokenizer vocab: ', len(tokenizer))
model = AutoModel.from_pretrained('xlm-roberta-base')
print('Embeddings size: ', model.embeddings.word_embeddings.num_embeddings)
model.resize_token_embeddings(len(tokenizer))
print('Embeddings size: ', model.embeddings.word_embeddings.num_embeddings)
tokenizer.save_pretrained('joel')
model.save_pretrained('joel')

"""
TODO Beschwerdeführer und Beschwerdegegner extrahieren aus Datenbank

maybe also it’s a good idea to just replace multiple defendants with ‘<defendant>  1’, ‘<defendant>  2’, instead of ‘<defendant_1>‘, ‘<defendant_2>’
i think it will be easier for the model to learn these new embeddings if they are repeated
And they have less tokens than <defendant> A._
and A,B,C,D seem to be completely not consistent
while def 1,2,3 is not
"""
