from numpy import *

def t(i,o,m,w):
 for j in range(m):
  p=1/(1+exp(-(dot(i,w))))
  e=o-p
  d=dot(i.T,e*(p*(1-p)))
  w+=d
 return w

# para entrenarla
w=t(array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]),array([[0,1,1,0]]).T,10000,random.random((3,1)))

# y que piense
print("-------------")
print("Tweet neural:")
print(w)
print(1/(1+exp(-dot([1,0,0],w))))
