# layout-ai

run "git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git"
than run "pip install unilm/layoutlm"

run "git clone https://github.com/huggingface/transformers.git"
than run "pip install transformers"

install req, to install torch go to https://pytorch.org/get-started/locally/

to run the training, first you need to make a "data" folder and in it a testing_data and a training_data folder when in  each one of them theres an annotations folder
in the annotations folder you'll add your annotations and when this is done, run "dataPrep.py",
this will do some changes to the folder "data", now youll need to create a "content" folder and a "data" folder in it, 
now run "training.py" and the training will start

file before running dataPrep.py:
layout-ai/
└─ data/
   ├─ training_data/
   └─ annotations/ {annotations.json}
   ├─ training_data/
   └─ annotations/ {annotations.json}
