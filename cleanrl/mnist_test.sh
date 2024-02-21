#!/bin/bash

# 파라미터별로 시도할 값들 설정
seeds=(0 1 2 3 4 5 6 7 8 9) # 예시로 0, 1, 2 사용
c_weights=(10 50 100) # 예시로 0.1, 0.01 사용
c_biases=(10 50 100) # 예시로 0.1, 0.01 사용
number_first_hiddens=(128 256 512) # 예시로 128, 256 사용
number_second_hiddens=(64 128 256)

# 모든 조합에 대해 스크립트 실행
for seed in "${seeds[@]}"; do
  for c_weight in "${c_weights[@]}"; do
    for c_bias in "${c_biases[@]}"; do
      for number_first_hidden in "${number_first_hiddens[@]}"; do
        for number_second_hidden in "${number_second_hiddens[@]}"; do
        echo "Running with seed=$seed, c_weight=$c_weight, c_bias=$c_bias, number_first_hidden=$number_first_hidden number_first_hidden=$number_second_hidden"
        python cleanrl/mnist_mlp.py --seed "$seed" --c_weight "$c_weight" --c_bias "$c_bias" --number_first_hidden "$number_first_hidden" --number_second_hidden "$number_second_hidden"
        done
      done
    done
  done
done