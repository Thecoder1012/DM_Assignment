import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("./chestXray14/Nih_All_d", output="./output", seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values
