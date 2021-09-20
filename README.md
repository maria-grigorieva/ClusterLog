# How to install clusterlogs module on lxplus
Firstly, connect to lxplus:

`ssh <USERNAME>@lxplus.cern.ch>`

Go to a place where you want to have the repo and clone it there:

https://github.com/micolocco/ClusterLog.git

After cloning the repo:

`cd ClusterLog`

Set up a new virtual environment there:

`python3 -m venv new_env`

Activate your virtualenv:

`source new_env/bin/activate`

Get kerberos ticket:

`kinit <username>@CERN.CH`

Setup environment to use HADOOP with the following commands (assuming that the user has access to the Analytix cluster):

`source "/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh"`

`source "/cvmfs/sft.cern.ch/lcg/etc/hadoop-confext/hadoop-swan-setconf.sh" analytix`

In this environment install the packages from requirements file,`clusterlogs` module and download dictionary required for pyTextRank library:

`python3 -m pip install --user -r requirements.txt`

`python3 -m spacy download --user en_core_web_sm `

`python3 -m pip  install --user clusterlogs`

done.

## After installation
Now your virtual env is set up correctly. Next time you access to lxplus follow these instructions:

`cd <yourfolder>`

`kinit <username>@CERN.CH`

`source new_env/bin/activate`

`source "/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh"`

`source "/cvmfs/sft.cern.ch/lcg/etc/hadoop-confext/hadoop-swan-setconf.sh" analytix`

## How to run training_corpus.py
After you followed the instructions described [After installation](##-after-installation) section, enter:

`python3 training_corpus.py <modelname>.model <number_of_days>`

ex:`python3 training_corpus.py test.model 3`
