'''
def total_euro(s, z):
    uk = sati*zarada
    print("Ukupno ", uk, " kesa")


sati = float(input("Unesi radne sate: "))
zarada = float(input("Unesi radne sate: "))
total_euro(sati, zarada)

p = 2
while p >1 or p<0:
    try:
        p = float(input("Ocjenu 0-1 "))
    except:
        print("Unos nije broj pokusaj ponovno")
    
if p <= 1 and p>=0.9:
    print("A")
elif p>0.9 and p>=0.8:
    print("B")
elif p>0.8 and p>=0.7:
    print("C")
elif p>0.7 and p>=0.6:
    print("D")
else:
    print("F")

inp = ""
nums = []
br = 0
while inp != "Done":
    inp = input("Unesi broj, za zavrs unesi \"Done\"")
    if inp.isdigit():
        nums.append(int(inp))
        br =+ 1
print("Uneseno brojeva ", br)
max = nums[0]
min = nums[0]
sum = 0

for num in nums:
    sum += num
    if num < min:
        min = num
    if num > max:
        max = num

print("Average: ", (sum/br))
print("Min: ", min)
print("Max: ", max)

nums.sort()
print("Sortirano", nums)
'''
'''
datName = input("Unesi ime dat: ")
f = open(datName, "r")
'''
br = 0


def word_count(string):
    br = dict()
    string = str(string)
    words = string.split(' ')
    for word in words:
        if word in br:
            br[word] += 1
        else:
            br[word] = 1
    return br


f = open("song.txt", "r")
# read all lines in a list
lines = f.readlines()
d = dict()
d = word_count(lines)
for key in d:
    if d[key] == 1:
        print("\"", key, "\"")
        br += 1

print(br)
