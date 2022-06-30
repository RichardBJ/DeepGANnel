# DeepGANnel
Our DeepGannel repository
This is the code that runs an example from Celik et al 2020 https://doi.org/10.1101/2020.06.25.171918
Git not fully ready yet. But this should work!
Note that there is code editing necessary, for example
(1) The location of your data
(2) The threshold you choose to determine which windows to include in training
(completely blank windows can be problematic).
(3) Optimal window length
Also, just testing it here and it hung the first 2 time for me because these files need to be there
(a) lr.txt  contains the learning rates. Just use ours as a template
(b) save.txt  simple textfile that has a boolean to save or not safe the created data.
(you don't want to save all the records whilst training is taking place!)
(c) Oh is a simple fix for this, but currently needs a folder called "images" available to save the output
as you go along.

rbj_gan_toon_Git1.ipynb runs two channel input....
Numan_conv1d_gan.py runs just for one channel (typically raw data)....

Funded by the BBSRC

