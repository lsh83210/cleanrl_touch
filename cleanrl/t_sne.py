import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
a=3
if a==1:
    Y=np.load('svd_scailing_action7.npy')

    X=np.load('svd_scailing_state7.npy')
    tsne = TSNE(n_components=2, random_state=0)
    # front_part_a = Y[:10000,:]
    front_part_s = X[:20000,:]
    u,sigma,v_trans=np.linalg.svd(front_part_s, full_matrices=True)
    squared_sigma = np.square(sigma)
    total_variance = np.sum(squared_sigma)
    # variance_ratios = squared_sigma / total_variance
    # v_trans=v_trans*(1.0/np.sqrt(variance_ratios))
    U, S, Vt = np.linalg.svd(v_trans.T)
    v_trans=np.dot(front_part_s,Vt)

    # middle_part=X[20000:30000,:]
    # next_part=X[150000:160000,:]
    back_part_a = Y[-80000:,:]
    # Ｙ = np.concatenate([front_part_ａ, back_part_ａ])
    back_part_s = X[-20000:,:]
    # colors = ['red' if value == 0 else 'blue' for value in back_part_a.flatten()] 
    Ｘ = np.concatenate([front_part_s, back_part_ｓ])
    X_2d = tsne.fit_transform(X)

    # num_points = X.shape[30000]  # 데이터 포인트의 총 수
    colors = np.linspace(0, 1, 20000)  # 각 10,000개마다 고유한 색상 값 할당
    colors = np.array(['blue' if i < 10000 else 'red' for i in range(X.shape[0])])
    # labels = np.repeat(colors,2)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='viridis', marker='.',s=1)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of state vectors')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig(f'action_space_svd.png')
else:
    X=np.load('svd_scailing_state7.npy')
    tsne = TSNE(n_components=2, random_state=0)
    front_part_s = X[:20000,:]
    back_part_s = X[-20000:,:]
    Ｘ = np.concatenate([front_part_s, back_part_ｓ])
    X_2d = tsne.fit_transform(X)
    colors = np.linspace(0, 1, 20000)  # 각 10,000개마다 고유한 색상 값 할당
    colors = np.array(['blue' if i < 10000 else 'red' for i in range(X.shape[0])])
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='viridis', marker='.',s=1)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of state vectors')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig(f'testing.png')