# 6.2 00:11 수정

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 모집단의 크기 n
population_size = 100000

# 모집단 1
u1, variance1 = map(float, input("모집단 1의 모평균, 모분산 입력: ").split())          # 모집단 1의 평균, 분산 입력

# 모집단 2
u2, variance2 = map(float, input("모집단 2의 모평균, 모분산 입력: ").split())          # 모집단 2의 평균, 분산 입력

# 두 개의 정규 분포로부터 데이터 생성
population1 = np.random.normal(u1, np.sqrt(variance1), population_size)    # 모집단 1에서 n의 개수만큼 Random Sampling한 표본집단1
population2 = np.random.normal(u2, np.sqrt(variance2), population_size)    # 모집단 2에서 n의 개수만큼 Random Sampling한 표본집단2

# 실제 데이터의 히스토그램 그리기
plt.hist(population1, bins=100, alpha=0.5, label='Population 1')
plt.hist(population2, bins=100, alpha=0.5, label='Population 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Populations')
plt.legend()
plt.show()

# 이론적인 확률 분포 그래프 그리기
x = np.linspace(min(population1.min(), population2.min()), max(population1.max(), population2.max()), 100)
pdf1 = (1 / np.sqrt(2 * np.pi * variance1)) * np.exp(-(x - u1)**2 / (2 * variance1))
pdf2 = (1 / np.sqrt(2 * np.pi * variance2)) * np.exp(-(x - u2)**2 / (2 * variance2))

plt.plot(x, pdf1, label='Population 1')
plt.plot(x, pdf2, label='Population 2')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Probability Density Functions')
plt.legend()
plt.show()

#문제 2번---------------------------------------------------------------------------------------------------------------------------------
sample_sizes = [10, 30, 61]

for sample_size in sample_sizes:

    # 두 개의 모집단으로부터 표본 추출
    sample1 = np.random.choice(population1, size=sample_size, replace=False)
    sample2 = np.random.choice(population2, size=sample_size, replace=False)

    # 표본 평균과 표본 분산 계산
    sample1_mean = np.mean(sample1)
    sample1_variance = np.var(sample1, ddof=1)
    sample2_mean = np.mean(sample2)
    sample2_variance = np.var(sample2, ddof=1)

    # 모분산이 같고 모분산을 알 때 모평균 추정 95% 신뢰구간 계산
    confidence_interval_mean1 = (sample1_mean - 1.96 * np.sqrt(variance1 / sample_size), sample1_mean + 1.96 * np.sqrt(variance1 / sample_size))
    confidence_interval_mean2 = (sample2_mean - 1.96 * np.sqrt(variance2/ sample_size), sample2_mean + 1.96 * np.sqrt(variance2 / sample_size))

    # 모분산 추정 95% 신뢰구간 계산
    chi_valueLeft = stats.chi2.ppf(0.975, sample_size - 1)  # 카이제곱 알파가 0.025 일 때
    chi_valueRight = stats.chi2.ppf(0.025, sample_size - 1)  # 카이제곱 알파가 0.975 일 때
    confidence_interval_variance1 = (((sample_size - 1) * sample1_variance ) / chi_valueLeft, ((sample_size - 1) * sample1_variance ) / chi_valueRight)
    confidence_interval_variance2 = (((sample_size - 1) * sample2_variance ) / chi_valueLeft, ((sample_size - 1) * sample2_variance ) / chi_valueRight)

    # 모분산을 모를 때 모평균 추정 95% 신뢰구간 계산
    t_value = stats.t.ppf(0.975, sample_size - 1)  # 95% 신뢰구간을 위한 T 값 (자유도에 따라 계산)
    T_confidence_interval1 = (sample1_mean - t_value * np.sqrt(sample1_variance / sample_size), sample1_mean + t_value * np.sqrt(sample1_variance / sample_size))
    T_confidence_interval2 = (sample2_mean - t_value * np.sqrt(sample2_variance/ sample_size), sample2_mean + t_value * np.sqrt(sample2_variance / sample_size))
    
    print(f"\n표본 크기가 {sample_size}일 때")
    print(f"모집단 1의 표본 평균: {sample1_mean:.3f}, 모집단 1의 표본 분산: {sample1_variance:.3f}")
    print(f"모집단 2의 표본 평균: {sample2_mean:.3f}, 모집단 2의 표본 분산: {sample2_variance:.3f}")
    print(f"모집단 1의 모평균 신뢰구간(Normal-dist.): P({confidence_interval_mean1[0]:.3f}" " < Population1 mean < " f"{confidence_interval_mean1[1]:.3f}) = 0.95" f"     간격: {(confidence_interval_mean1[1] - confidence_interval_mean1[0]):.3f}")
    print(f"모집단 2의 모평균 신뢰구간(Normal-dist.): P({confidence_interval_mean2[0]:.3f}" " < Population2 mean < " f"{confidence_interval_mean2[1]:.3f}) = 0.95" f"      간격: {(confidence_interval_mean2[1] - confidence_interval_mean2[0]):.3f}")
    print(f"모집단 1의 모분산 신뢰구간(Chi-dist.):    P({confidence_interval_variance1[0]:.3f}" " < Population1 variance < " f"{confidence_interval_variance1[1]:.3f}) = 0.95" f"   간격: {(confidence_interval_variance1[1] - confidence_interval_variance1[0]):.3f}")
    print(f"모집단 2의 모분산 신뢰구간(Chi-dist.):    P({confidence_interval_variance2[0]:.3f}" " < Population2 variance < " f"{confidence_interval_variance2[1]:.3f}) = 0.95" f"   간격: {(confidence_interval_variance2[1] - confidence_interval_variance2[0]):.3f}")
    print(f"모집단 1의 모평균 신뢰구간(T-dist.):      P({T_confidence_interval1[0]:.3f}" " < Population1 mean < " f"{T_confidence_interval1[1]:.3f}) = 0.95" f"     간격: {(T_confidence_interval1[1] - T_confidence_interval1[0]):.3f}")
    print(f"모집단 2의 모평균 신뢰구간(T-dist.):      P({T_confidence_interval2[0]:.3f}" " < Population2 mean < " f"{T_confidence_interval2[1]:.3f}) = 0.95" f"      간격: {(T_confidence_interval2[1] - T_confidence_interval2[0]):.3f}")

#문제 3번---------------------------------------------------------------------------------------------------------------------------------#문제 3번---------------------------------------------------------------------------------------------------------------------------------
sample_sizes = [81, 101]

for sample_size in sample_sizes:
    # 두 개의 모집단으로부터 샘플 추출
    sample1 = np.random.choice(population1, size=sample_size, replace=False)
    sample2 = np.random.choice(population2, size=sample_size, replace=False)

    # 두 표본 평균 계산
    sample1_mean = np.mean(sample1)
    sample2_mean = np.mean(sample2)

    # 두 표본 분산 계산
    sample1_variance = np.var(sample1, ddof=1)
    sample2_variance = np.var(sample2, ddof=1)

    # 신뢰구간 계산 assuming we know the population variance
    # Z0.025 = 1.96
    confidence_interval1 = ((sample1_mean - sample2_mean) - 1.96 * np.sqrt((variance1 / sample_size) + (variance2 / sample_size)),
                        (sample1_mean - sample2_mean) + 1.96 * np.sqrt((variance1 / sample_size) + (variance2 / sample_size)))

    # 신뢰구간 계산 assuming we don't know the population variance but we know both are the same
    # 합동추정량 Sp**2
    pooled_variance = ((sample_size - 1) * sample1_variance + (sample_size - 1) * sample2_variance) / (sample_size + sample_size - 2)
    t_value1 = stats.t.ppf(0.975, sample_size + sample_size - 2)  # 95% 신뢰구간을 위한 T 값 (자유도에 따라 계산)
    confidence_interval2 = ((sample1_mean - sample2_mean) - t_value1 * np.sqrt(pooled_variance * (1/sample_size + 1/sample_size)),
                        (sample1_mean - sample2_mean) + t_value1 * np.sqrt(pooled_variance * (1/sample_size + 1/sample_size)))

    # 신뢰구간 계산 assuming we don't know the population variance
    v = ((sample1_variance / sample_size + sample2_variance / sample_size)**2) / (((sample1_variance / sample_size)**2 / (sample_size - 1)) + ((sample2_variance / sample_size)**2 / (sample_size - 1)))
    t_value2 = stats.t.ppf(0.975, v)  # 95% 신뢰구간을 위한 T 값 (자유도에 따라 계산)
    confidence_interval3 = ((sample1_mean - sample2_mean) - t_value2 * np.sqrt((sample1_variance / sample_size) + (sample2_variance / sample_size)),
                        (sample1_mean - sample2_mean) + t_value2 * np.sqrt((sample1_variance / sample_size) + (sample2_variance / sample_size)))

    print(f"\n표본 크기가 {sample_size}일 때")
    print(f"모집단 1의 표본 평균: {sample1_mean:.3f}")
    print(f"모집단 2의 표본 평균: {sample2_mean:.3f}")
    print(f"두 모집단의 평균의 차의 신뢰구간(Normal-dist.):              P({confidence_interval1[0]:.3f}" " < Population1 mean - Population2 mean < " f"{confidence_interval1[1]:.3f}) = 0.95" f" 간격: {(confidence_interval1[1] - confidence_interval1[0]):.3f}")
    print(f"두 모집단의 평균의 차의 신뢰구간(T-dist. | 합동추정량 이용): P({confidence_interval2[0]:.3f}" " < Population1 mean - Population2 mean < " f"{confidence_interval2[1]:.3f}) = 0.95" f" 간격: {(confidence_interval2[1] - confidence_interval2[0]):.3f}")
    print(f"두 모집단의 평균의 차의 신뢰구간(T-dist.):                   P({confidence_interval3[0]:.3f}" " < Population1 mean - Population2 mean < " f"{confidence_interval3[1]:.3f}) = 0.95" f" 간격: {(confidence_interval3[1] - confidence_interval3[0]):.3f}\n")
