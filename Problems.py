
import random
random.seed(int(input()))

total_winnings = 0  # Initialize total winnings
while True:
   bet_input = input()
   if bet_input == "DONE":
      break
   else:
      bet = int(bet_input)
      slots = [random.randint(1, 7) for _ in range(3)]  # Generate 3 random numbers
      print("Bet:", end=' ')
      for i in slots:
         print(i, end=' ')
      if all(num == 7 for num in slots):
         winnings = bet * 7
      elif len(set(slots)) == 1:
         winnings = bet * 3
      elif slots[0] == slots[1] or slots[1] == slots[2]:
         winnings = bet * 2
      else:
         winnings = -bet
      total_winnings += winnings
      print(winnings)
print("Bet:", total_winnings)


# count punctuation
sentance = input()
count = 0
for c in sentance:
   if not (c.isalnum() or c.isspace()):
      count += 1 
print(count)

# is_prime (201 M5)
def is_prime(n):
    if n == 1:
      return False
    for i in range(2, int(n**0.5)+1):
      if n % i == 0:
         return False
    return True

# compare last bit 
input_s = input().split()
num_1 = int(input_s[0])
num_2 = int(input_s[1])
if (num_1 % 2 == 0 and  num_2 % 2 == 0) or (num_1 % 2 == 1 and  num_2 % 2 == 1):
   print("True")
else:
   print("False")


# string practice 
str_inp = input()


# In the first line, print the third character of this string.
print(str_inp[2])
# In the second line, print the second to last character of this string.
print(str_inp[-2:-1])
# In the third line, print the first five characters of this string.
print(str_inp[:5])
# In the fourth line, print all but the last two characters of this string.
print(str_inp[:len(str_inp)-2])
# In the fifth line, print all the characters of this string with even indices 
#(remember indexing starts at 0, so the characters are displayed starting with the first).
print(str_inp[::2])
# In the sixth line, print all the characters of this string with odd indices 
#(i.e. starting with the second character in the string).
print(str_inp[1::2])
# In the seventh line, print all the characters of the string in reverse order.
print(str_inp[::-1])
# In the eighth line, print every second character of the string in reverse order, 
# starting from the last one.
print(str_inp[-1::-2])
# In the ninth line, print the length of the given string.
print(len(str_inp))

#merge  intervals
def merge_intervals(intervals):
   i = 0
   while i < len(intervals)-1:
      if intervals[i][1] >= intervals[i+1][0]:
         intervals[i][1] = max(intervals[i][1],intervals[i+1][1])
         intervals.pop(i+1)
      else:
         i +=1
   return intervals

# sets
set_1 = set(input().split())
set_2 = set(input().split())

result = list(set_1.intersection(set_2))
result = sorted(map(int, result))
for num in result:
  print(num, end = " ")

# see number for the first or second time
num_input = [int(x) for x in input().split()]
num_set = set()
for num in num_input:
   if num in num_set:
      print("YES")
   else:
      print("NO")
      num_set.add(num)

# dict: word frequency 
num_of_lists = int(input())
word_number = dict()
for i in range(num_of_lists):
   words_in_line = input().split()
   for word in words_in_line:
      if word in word_number:
         word_number[word] += 1
      else:
         word_number[word] = 1
values = list(word_number.values())
same_freq = [word for word, freq in word_number.items() if freq == max(values) ]
same_freq.sort()
print(same_freq[0])

# dict: find synonyms
size_of_dict = int(input())
synonyms = dict()
for i in range(size_of_dict):
   line = input().split()
   synonyms[line[0]] = line[1]
word = input()
keys = list(synonyms.keys())
values = list(synonyms.values())

if word in synonyms.keys():
   print(values[keys.index(word)])
else: 
   print(keys[values.index(word)])

# dict: commands
size = int(input())
file_permissions = {} 
ops_descr = {"execute":"X", "write": "W", "read" : "R"}
for _ in range(size):
   file_name, *allowed_ops = input().split()
   file_permissions[file_name] = set(allowed_ops)

M = int(input()) 
for _ in range(M):
   requested_op, file_name = input().split()
   ops_code = ops_descr[requested_op]
   if file_name in file_permissions and ops_code in file_permissions[file_name]:
      print("OK")
   else:
      print("Access denied")

#dict: word freq 
word_input = list(input().split())
word_freq = {}
output = []

for word in word_input:
   if word not in word_freq.keys():
      output.append(0)
      word_freq[word] = 1
   else:
      output.append(word_freq[word])
      word_freq[word] += 1
for _ in output:
   print(_, end=" ")

# count primes
def countPrimes(self, n: int) -> int:
        if n<=2:
            return 0
        ref=[True]*(n)
        i=2
        while (i*i)<n:
            if ref[i]:
                for j in range(i*i,n,i):
                    ref[j]=False
            i+=1
        return ref.count(True)-2

# last digit of the factorial
n = int(input())

if n == 0 or n == 1:
   print(1)
else:
   result = 1
   for i in range(1,n+1):
      result *= i
   print(result%10)

# trailing 0s approach after n > 5
n = int(input())

result = 1
if n < 6:
   
   for i in range(1, n+1):
      result *= i
   result %= 10
else:
   result = 0
print(result)


# first digit in factorial
n = int(input())

result = 1
if n != 0 or n != 1:
   for i in range (1, n+1):
	   result *= i
print(str(result)[0])

# last non 0 in factorial
n = int(input())

result = 1
if n != 0 or n != 1:
   for i in range (1, n+1):
      result *= i
   while result%10 == 0:
      result = result//10
print(result%10)