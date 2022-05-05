import numpy as np

def create_missingness(tensor, drop):
    """
    Creates missingness for a full tensor. Slgihtly faster than entry_drop()
    """
    idxs = np.argwhere(np.isfinite(tensor))
    dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
    dropidxs = tuple(dropidxs.T)
    tensor[dropidxs] = np.nan


def entry_drop(tensor, drop, seed=None):
    """
    Drops random values within a tensor. Finds a bare minimum cube before dropping values to ensure PCA remains viable.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape. Preference for at least two values present per chord.
    drop : int
        To set a percentage, multiply np.sum(np.isfinite(tensor)) by the percentage
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    None : tensor is modified with missing values.
    """
    # Track chords for each mode to ensure bare minimum cube covers each chord at least once

    if seed != None:
        np.random.seed(seed)

    midxs = np.zeros((tensor.ndim, max(tensor.shape)))
    for i in range(tensor.ndim):
        midxs[i] = [1 for n in range(tensor.shape[i])] + [0 for m in range(len(midxs[i]) - tensor.shape[i])]
    modecounter = np.arange(tensor.ndim)

    # Remove bare minimum cube idxs from droppable values
    idxs = np.argwhere(np.isfinite(tensor))
    while np.sum(midxs) > 0:
        removable = False
        ran = np.random.choice(idxs.shape[0], 1)
        ranidx = idxs[ran][0]
        counter = 0
        for i in ranidx:
            if midxs[modecounter[counter], i] > 0:
                removable = True
            midxs[modecounter[counter], i] = 0
            counter += 1
        if removable == True:
            idxs = np.delete(idxs, ran, axis=0)
    assert idxs.shape[0] >= drop

    # Drop values
    dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
    dropidxs = [tuple(dropidxs[i]) for i in range(drop)]
    for i in dropidxs: tensor[i] = np.nan


def joint_entry_drop(big_tensor, small_tensor, drop, seed=None):

    if seed != None:
        np.random.seed(seed)
    
    # Track chords for each mode to ensure bare minimum cube covers each chord at least once
    midxs1 = np.zeros((big_tensor.ndim, max(big_tensor.shape)))
    for i in range(big_tensor.ndim):
        midxs1[i] = [1 for n in range(big_tensor.shape[i])] + [0 for m in range(len(midxs1[i]) - big_tensor.shape[i])]
    modecounter1 = np.arange(big_tensor.ndim)

    midxs2 = np.zeros((small_tensor.ndim, max(small_tensor.shape)))
    for i in range(small_tensor.ndim):
        midxs2[i] = [1 for n in range(small_tensor.shape[i])] + [0 for m in
                                                                 range(len(midxs2[i]) - small_tensor.shape[i])]
    modecounter2 = np.arange(small_tensor.ndim)

    # Remove bare minimum cube idxs from droppable values
    idxs1 = np.argwhere(np.isfinite(big_tensor))
    while np.sum(midxs1) > 0:
        removable = False
        ran = np.random.choice(idxs1.shape[0], 1)
        ranidx = idxs1[ran][0]
        counter = 0
        for i in ranidx:
            if midxs1[modecounter1[counter], i] > 0:
                removable = True
            midxs1[modecounter1[counter], i] = 0
            counter += 1
        if removable == True:
            idxs1 = np.delete(idxs1, ran, axis=0)

    idxs2 = np.argwhere(np.isfinite(small_tensor))
    while np.sum(midxs2) > 0:
        removable = False
        ran = np.random.choice(idxs2.shape[0], 1)
        ranidx = idxs2[ran][0]
        counter = 0
        for i in ranidx:
            if midxs2[modecounter2[counter], i] > 0:
                removable = True
            midxs2[modecounter2[counter], i] = 0
            counter += 1
        if removable == True:
            idxs2 = np.delete(idxs2, ran, axis=0)

    # Combine droppable idxs
    temp1 = np.hstack((np.full((idxs1.shape[0], 1), 1), idxs1))
    temp2 = np.hstack((np.full((idxs2.shape[0], 1), 2), idxs2))
    diff = idxs1.shape[1] - idxs2.shape[1]
    for _ in range(diff):
        temp2 = np.hstack((temp2, np.full((temp2.shape[0], 1), 0)))
    joint_idxs = np.vstack((temp1, temp2))
    assert joint_idxs.shape[0] >= drop

    # Select dropped idxs
    dropidxs = joint_idxs[np.random.choice(joint_idxs.shape[0], drop, replace=False)]

    # Separate dropped idxs and drop in corresponding tensors
    tensor1_dropped, tensor2_dropped = 0, 0
    for i in range(dropidxs.shape[0]):
        if dropidxs[i, 0] == 1:
            tensor1_dropped += 1
        if dropidxs[i, 0] == 2:
            tensor2_dropped += 1

    if tensor1_dropped > 0:
        dropidxs1 = np.zeros((tensor1_dropped, dropidxs.shape[1]), dtype=int)
        counter = 0
        for i in range(dropidxs.shape[0]):
            if dropidxs[i, 0] == 1:
                dropidxs1[counter] = dropidxs[i]
                counter += 1
        dropidxs1 = np.delete(dropidxs1, 0, 1)
        dropidxs1 = [tuple(dropidxs1[i]) for i in range(tensor1_dropped)]
        for i in dropidxs1: big_tensor[i] = np.nan
    if tensor2_dropped > 0:
        dropidxs2 = np.zeros((tensor2_dropped, dropidxs.shape[1]), dtype=int)
        counter = 0
        for i in range(dropidxs.shape[0]):
            if dropidxs[i, 0] == 2:
                dropidxs2[counter] = dropidxs[i]
                counter += 1
        dropidxs2 = np.delete(dropidxs2, 0, 1)
        for i in range(diff): dropidxs2 = np.delete(dropidxs2, -1, 1)
        dropidxs2 = [tuple(dropidxs2[i]) for i in range(tensor2_dropped)]
        for i in dropidxs2: small_tensor[i] = np.nan


def chord_drop(tensor, drop, seed=None):
    """
    Removes chords along axis = 0 of a tensor.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape.
    drop : int
        To set a percentage, multiply tensor.shape[0] by the percentage
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    None : tensor is modified with missing chords.
    """

    if seed != None:
        np.random.seed(seed)

    # Drop chords based on random idxs
    chordlen = tensor.shape[0]
    for _ in range(drop):
        idxs = np.argwhere(np.isfinite(tensor))
        chordidx = np.delete(idxs[np.random.choice(idxs.shape[0], 1)][0], 0, -1)
        dropidxs = []
        for i in range(chordlen):
            dropidxs.append(tuple(np.insert(chordidx, 0, i).T))
        for i in range(chordlen):
            tensor[dropidxs[i]] = np.nan

