import torch

# 행렬의 크기를 정의
rows = 5
cols = 5
number=50
# 모든 요소가 0인 초기 행렬 생성
matrix = torch.zeros(rows*number, cols)

# 각 행에 대해 1의 위치를 랜덤하게 선택
indices = torch.randint(0, cols, (number*rows,))

# 선택된 위치에 1 할당
matrix[torch.arange(rows*number), indices] = 1
matrix=matrix.view(number,rows,cols)
print(matrix)