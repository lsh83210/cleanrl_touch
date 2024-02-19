import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


randoms=np.random.normal(loc=0, scale=1, size=(10000, 20))
# breakpoint()
# U, S, VT = np.linalg.svd(randoms)
# print(S)
# X=np.load('datas.npy')
tsne = TSNE(n_components=2, random_state=0)
# front_part = X[:10000,:]
# middle_part=X[20000:30000,:]
# next_part=X[150000:200000,:]
# back_part = X[-10000:,:]
# X = np.concatenate([front_part,middle_part,next_part, back_part])
X_2d = tsne.fit_transform(randoms)
# # num_points = X.shape[30000]  # 데이터 포인트의 총 수
colors = np.linspace(0, 1, 300)  # 각 10,000개마다 고유한 색상 값 할당
# labels = np.repeat(colors, 10000)
# fig=plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d') 
# ax.scatter(X_2d [:, 0], X_2d [:, 1], X_2d [:, 2])
# ax.set_title('3D t-SNE Visualization')
# ax.set_xlabel('t-SNE 1')
# ax.set_ylabel('t-SNE 2')
# ax.set_zlabel('t-SNE 3')
# plt.show()
colors = np.linspace(0, 1, 10000)  # 각 10,000개마다 고유한 색상 값 할당
# colors = np.array(['blue' if i < 10000 else 'red' for i in range(X.shape[0])])
# labels = np.repeat(colors,2)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],c=colors, cmap='viridis', marker='.',s=10)
plt.colorbar(scatter)
plt.title('t-SNE visualization of state vectors')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.savefig(f'gaussian_space.png')