import sys
sys.path.insert(0,'/u/f/fbarone/Documents/TWO-NN')
import os
import numpy as np
import matplotlib.pyplot as plt
import id2nn



# Decoder block analysis

for act in os.listdir('./activations'):
    act_path = os.path.join('activations', act)
    print(act_path)
    with open(act_path, 'rb') as f:
        features = np.load(f)

    np.random.seed(10)

    # dim, mu, mu_cs = two_nn_id(data, 0.9)

    blocks_dim, blocks_dim_std, blocks_size, d_mat2 = id2nn.two_nn_block_analysis(features, .9, shuffle = True)

    # print(len(blocks_dim))
    # # print(len(blocks_dim_std))

    block_path = os.path.join('activations', 'block_' + act)
    print(block_path)
    with open(block_path, 'w') as file: 
        blocks_dim = np.array(blocks_dim)[:,None]
        blocks_dim_std = np.array(blocks_dim_std)[:,None]
        blocks = np.hstack((blocks_dim, blocks_dim_std))
        np.savetxt(file, blocks)


# Encoder block activation

for act in os.listdir('/scratch/fbarone/activations_CAE_PCA-1024'):
    act_path = os.path.join('activations', act)
    print(act_path)
    with open(act_path, 'rb') as f:
        block_dim = np.load(f)
        block_std = np.load(f)
        

    np.random.seed(10)

    # dim, mu, mu_cs = two_nn_id(data, 0.9)

    blocks_dim, blocks_dim_std, blocks_size, d_mat2 = id2nn.two_nn_block_analysis(features, .9, shuffle = True)

    # print(len(blocks_dim))
    # # print(len(blocks_dim_std))

    block_path = os.path.join('activations', 'block_' + act)
    print(block_path)
    with open(block_path, 'w') as file: 
        blocks_dim = np.array(blocks_dim)[:,None]
        blocks_dim_std = np.array(blocks_dim_std)[:,None]
        blocks = np.hstack((blocks_dim, blocks_dim_std))
        np.savetxt(file, blocks)



fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
for act in os.listdir('./activations'):
    if 'block' in act:
        with open('./activations/' + act, 'r') as f:
            blocks = np.loadtxt(f)
#             ax.plot(blocks_size, blocks[:,0], "r.-")
            ax.errorbar(blocks_size, blocks[:,0], yerr = np.array(blocks[:,1]), label = act)
    
ax.legend()



# fig_path = os.path.join( 'block_analysis.png' )
# fig.savefig(fig_path, bbox_inches='tight')
