from logNormal import LogNormal
from blackscholes import BlackScholes


m = LogNormal()

m.graficar(m.em,3)


s0 = 74.625
K = 100
T = 1.6
r = 0.05
sigma = 0.375


f = BlackScholes(s0,K,T,r,sigma)
print(f)
