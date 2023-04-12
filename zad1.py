
import pandas as pd

mtcars = pd.read_csv('mtcars.csv')
print(mtcars)

#1
print('\n\nSLIJEDECI ZADATAK\n\n')
potrosnjaSortirani = mtcars.sort_values(by = ['mpg'],ascending=False).head(5)
print(potrosnjaSortirani[['car','mpg']])

#2
print('\n\nSLIJEDECI ZADATAK\n\n')
auto3minPotrosnja8cil = mtcars[mtcars['cyl']==8].sort_values(by='mpg',ascending=True).tail(3)
print(auto3minPotrosnja8cil[['car','cyl','mpg']])

#3
print('\n\nSLIJEDECI ZADATAK\n\n')

autoAVG6cyl = mtcars[mtcars['cyl']==6]['mpg'].mean()
print(autoAVG6cyl)

#4
print('\n\nSLIJEDECI ZADATAK\n\n')
autoAVG4cylWT2k__2_2k = mtcars[(mtcars['cyl']==4) & ((mtcars['wt']>2.000) & (mtcars['wt']<2.200))]['mpg'].mean()
print(autoAVG4cylWT2k__2_2k)

#5
print('\n\nSLIJEDECI ZADATAK\n\n')
print('AUTO MJENJAC ' + str(mtcars[(mtcars['am'] == 1)]['am'].count()))

print('EMANUEL MJENJAC ' + str(mtcars[(mtcars['am'] == 0)]['am'].count()))

#6
print('\n\nSLIJEDECI ZADATAK\n\n')

print('AUTO MJENJAC 100+ hp : ' + str(mtcars[(mtcars.am == 1 ) & (mtcars.hp>100)]['hp'].count()))

#7
print('\n\nSLIJEDECI ZADATAK\n\n')
lbsKgConst = 0.45359237

print(mtcars[['car','wt']])
stupacKile = mtcars['wt'] * lbsKgConst
kileAuta = pd.concat([mtcars['car'], stupacKile], axis=1)
print(kileAuta[['car','wt']])