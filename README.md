# DeepGANnel
Our DeepGANnel repository
This is the code that runs an example from [Ball et al 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267452)
Data also on FigShare: https://figshare.com/s/fe0eb801ed18691439ea
Git not fully ready yet. But this should work!

rbj_gan_toon_Git1.ipynb runs two channel input....
Numan_conv1d_gan.py runs just for one channel (typically raw data).... filder "Raw_data_only_GAN"

In general, synthetic output files have "gen" in the title and raw input has the word raw in it.  Perhaps the figshare above is a simpler format to look at since it doesn't have all ther revisions.

Note that we ran each of these models/with different datasets in different branches.  We were tempted to merge these, but actually now have left in the diffferent branches since merging brings with it inherent conflicts.  Differences between the different branches include different datasets, different record lengths and minor changes to the hyperparameters.  Hyperparameters for any given run are now automatically saved to file.
NOTE the embedding figures were created here: https://github.com/RichardBJ/GANEmbedding
And this includes the final source and test datasets a little more neatly!

Funded by the BBSRC

