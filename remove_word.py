# remove the "train_" or "test_" at the beginning of the variables
def remove(word):
    if word.startswith("train_"):
        return word[6:]
    elif word.startswith("test_"):
        return word[5:]
    else:
        return word