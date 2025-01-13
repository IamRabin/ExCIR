import numpy as np
import shutil
import torch



def orthonormal_space(data,n_eig_vecs):
    """
    Calculate the orthonormal space of a data matrix using SVD.

    Args:
        data (numpy.ndarray): The data matrix, of shape (n, k).
        n_eig_vecs (int): The number of eigenvectors to keep. If None, all eigenvectors will be kept.

    Returns:
        numpy.ndarray: The orthonormal space, of shape (n, n_eig_vecs).
    """

    n, k = data.shape
    u, s, v = np.linalg.svd(data)


    orthonormal_space = u[:, :n_eig_vecs] 

    return torch.tensor(orthonormal_space, dtype=torch.float32)



    # sample from X and y 

def sample_data(X, y, num_samples=250):
    num_data = len(X)
    indices = np.random.choice(num_data, size=num_samples, replace=False)
    sampled_X = X[indices]
    sampled_y = y[indices]
    return sampled_X, sampled_y


def sample_vector(M):
    """
    Samples a random vector from the space spanned by the columns of the orthonormal matrix M.
    """
    # Generate a random vector of coefficients
    coeffs = np.random.randn(M.shape[1])
    # Sample a vector from the space spanned by M
    vector = M @ coeffs
    return vector

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc






def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_acc_max = checkpoint['valid_acc_max']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_acc_max




