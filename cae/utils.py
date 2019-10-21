import numpy as np
import imageio

# based on https://github.com/foamliu/Autoencoder/blob/master/utils.py
class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.val = 0
        self.avg = 0
        self.count = 0
        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.val + (1 - self.alpha) * self.avg


def read_image_png(file_name):
    """
    Auxiliary loader function to create the dataset using 
    """
    image = np.array(imageio.imread(file_name)).astype(np.int32)
    return image


# ##############################################
# Activations

def get_activation(layer_dict, name):
    """
    Define hook to extract intermediate layer features.

    :param layer_dict: dictionary with the activations per layer.
    :param name: specific layer name

    """
    def hook(model, input, output):
        layer_dict[name].append(output.detach()) #.view((output.shape[0],-1)))
        # layer_dict[name] = torch.cat(layer_dict[name], output.detach())
    return hook


def save_activations(activations_dict, output_folder):
    """
    Save into separate files activations extracted with a hook. 
    One file per layer/activation.
    :param layer_dict: dictionary with the activations per layer.
    :param output_data_folder: path to folder where to store the activation files.
    """
    
    if os.path.exists(output_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save the activations already exists.")
        return
    else:
        os.makedirs(output_folder)


    for name, activation in activations_dict.items():
        file_path = os.path.join(output_folder,name + '.pkl')
        print(file_path)
        with open(file_path,'wb') as file: 
            np.save(file, activation.numpy())
            # np.savetxt(file, activation.numpy())


def load_activations(input_folder):
    """
    Load activations from different files into a single dictionary. 
    One file per layer/activation.
    :param layer_dict: dictionary with the activations per layer.
    :param output_data_folder: path to folder where to store the activation files.
    """
    activations = {}
    for file in os.listdir(input_folder):
        if file.endswith('.pkl'):
            full_path = os.path.join(input_folder, file)
            name = file[:-4]
            activation = load_single_activation(full_path)
            activations[name] = activation
    return activations


    
def load_single_activation(input_file):
    """
    Load single activation from a binary file (saved as pickle)
    """
    with open(input_file, 'rb') as file:
        return np.load(file)

# ##############################################