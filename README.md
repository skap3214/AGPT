# AGPT

## Goal
I want to create a LLM from the beginning.
My goal for this project is to simultaneously learn and develop an LLM by myself.
There is only one constraint: I should understand everything that I implement.

So far this is a simple decoder-only model which uses masked multi head attention to generate responses. I have followed along the Karpathy makemore series to create this teeny tiny model.

## Model
UPDATE: AGPT_large_v3 has been trained. It has around 45M parameters and was trained on 1 A100 GPU and does output coherent sentences! In the process of training a larger model.

Here is a snippet of the response (max tokens set to 128 tokens):

Input: Once upon a time

Response (input concatenated with reponse):
 Once upon a time Jacky was very excited because today was was going on a special safari with her mom . It was a big , │
│ modern safari , shiny red and she couldn ' t wait to see the animals . So she asked the mom if they could see the     │
│ lions , the elephants and go . They all thought it would be fun to see all the animals away . Ella started to feel    │
│ very noisy , but she knew she had to leave . She asked her mom if they could stay out of their car and never come on  │
│ an adventure . They drove for a few hours and Max ran towards the ostrich and Ella followed the ostrich with it . He  │
│ felt very uncomfortable and weak . He was brought some snacks and drinks with her on it . Then he promised his        │
│ favorite animals they would come over again soon . The ostrich was so thankful for her adventure and the two animals  │
│ full of joy to see the animals in the jungle . They saw elephants , lions , tigers and elephants happily under the    │
│ monkeys . They had so much fun ! From then on , the elephant and Ella shared treats , swimming and knew that no       │
│ matter what the  

Much better than the previous model!:

Input: center a div

Response (input concatenated with reponse):
center a divis sal rorere ef to soss mipbin oa contsinmato oron ecit reclk thebeto th ur this ond. dent ais whand f whoun] secr can pd oxtiches ther soj or a l  oa sok pri thirein ofdemicor tiop secyou co'p a lo i.J to ohes the yosos es of oree o othor sis bo'kan simiry pisror in tirone yos. Goud St smealeve te ' she.e. tor sag baslaconshat cand orexts,1 thtkert orlior dhe a ser+.
T. ` wicul this bo, to no ches on cher in the l cor to to sfofou thev, yol cawe and ca oyound to sertroc r wac dand `kOs rteecbe

## Details
Each model version has these files in its corresponding folder:
  - config.py contains the hyperparameters of the model
  - model.py contains the code for the final model
  - training.py contains all the training code for the model.
  - generate.py contains the code for generating responses from the model
  - tokenizer.py contains the code to train the tokenizer for the model

## Dataset
I am currently using datasets/tiny_stories_full.txt which I found from huggingface. I plan to use a med dataset for the next model I train.

## References/Tutorials
List of all the resources I used to build this:
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html#functions-to-generate-input-and-target-sequence
- https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- https://github.com/mistralai/mistral-src
- https://blog.briankitano.com/llama-from-scratch/
