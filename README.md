# Abstract
 This thesis work develops within the WLCG Operational Intelligence ([OpInt](https://operational-intelligence.web.cern.ch/)) project. OpInt is born as a common project among the Worldwide LHC Computing Grid ([WLCG](https://wlcg.web.cern.ch/)) communities with the purpose of creating solutions to increase the level of automation in computing operations and reduce human interventions.  Indeed the currently deployed systems have been successful in satisfying the experiment goals; however, the High Luminosity LHC era will significantly increase the volume of data around the world therefore requiring to automate the computing operations in order to efficiently manage the
computing infrastructures. 

The system responsible for file transfer across the LHC Computing Grid is FTS. The File Transfer Service ([FTS](https://fts.web.cern.ch/fts/)) allows to sustain a data transfer rate of 20-40 GB/s, and it
transfers daily a few millions files. When a transfer fails, it produces an error message that is managed by the system allowing to reach the shifters. The number of error
messages per day to be monitored is of the order of a few hundred thousand. It is in this context that we look at Machine Learning as a tool to develop smart solutions
and optimize our resources. 

In particular, in this thesis, we apply Natural Language Processing techniques to analyze FTS error messages. **The aim of the work is to group the error messages
into meaningful clusters in order to trustingly speed up the error detection process.**


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

# Analysis schema
![schema](analysis_schema.PNG)
