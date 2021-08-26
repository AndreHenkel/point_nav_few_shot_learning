

# installation
install conda

create new environment with:
$conda create -n MAML_replication_study python=3.7 anaconda

$source activate MAML_replication_study


#Go in the point-nav directory and do:

$pip install -e .

#Then go in the MAML_replication_study directory and do:

$pip install -r requirements.txt


# training

$python train.py

# testing

$python test.py

# display

$python show.py

#you can also train the basic comparison of a normal pretrained model with:
$python basic_train_comparison.py

#Parameters and models need to be changed within the scripts.

# packages:
torch==1.3.0
gym
ptan


# contact
For any questions, let me know:
andre.henkel@uni-ulm.de


