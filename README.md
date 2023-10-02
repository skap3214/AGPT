# AGPT

## Goal
Creating and developing an LLM in only PyTorch.
My goal for this project is to simultaneously learn and develop an LLM using only PyTorch.

So far this is a simple decoder-only model which uses masked multi head attention to generate responses. I have followed along the Karpathy makemore series to create this teeny tiny model.

## Model
UPDATE: AGPT_small has been trained. It has 73502 params(very small) and does not output coherent sentences! In the process of training a larger model.
It is very overfitted since I am running all this on a macbook. I am in the process of training the large model with ~11m parameters.

Here is a snippet of the response (max tokens set to 500):

(Sorry for the weird format)

Question: center a div

==========Model Response==========
center a divis sal rorere ef to soss mipbin oa contsinmato oron ecit reclk thebeto th ur this ond. dent ais whand f whoun] secr can pd oxtiches ther soj or a l  oa sok pri thirein ofdemicor tiop secyou co'p a lo i.J to ohes the yosos es of oree o othor sis bo'kan simiry pisror in tirone yos. Goud St smealeve te ' she.e. tor sag baslaconshat cand orexts,1 thtkert orlior dhe a ser+.
T. ` wicul this bo, to no ches on cher in the l cor to to sfofou thev, yol cawe and ca oyound to sertroc r wac dand `kOs rteecbe

## Details
config.py contains the hyperparameters of the model
model.py contains the models and helper functions(which I will add to a seperate file later)
training.py contains all the training code for the model.
generate.py contains the code for generating responses from the model
## Dataset
I am currently using ./datasets/qa_conversations.txt which I found from huggingface
