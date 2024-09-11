
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

#trailing 0s in factorial
def trailingZeroes(self, n: int) -> int:
   zero_count = 0
   for i in range(5, n + 1, 5):
      current = i
      while current % 5 == 0:
         zero_count += 1
         current //= 5

   return zero_count

#missing int

def first_missing_int(integers):
 integers.sort()
 for i in integers:
   if i != integers.index(i):
      return integers.index(i)
 return len(integers)

# missing positive integer unsorted input
 def missing_positive_int_unsorted(integers):
   sumOfNums = 0
   for i in integers:
    sumOfNums += i
   return len(integers)*(len(integers) +1)//2 - sumOfNums

# largest common divisor 
m = int(input())
n = int(input())

if n == 0 or m == 0:
   print(max(m,n))
else:
   maxDiv = 0
   i = 1
   while i <= min(m,n):
      if n % i == 0 and m % i == 0:
         maxDiv = i
      i+=1
   print(maxDiv) 

#better approach(from the end)
m = int(input())
n = int(input())

if n == 0 or m == 0:
   print(max(m,n))
else:
   maxDiv = min(m,n)
   while maxDiv >= 1:
      if n % maxDiv == 0 and m % maxDiv == 0:
         print(maxDiv)
         break
      maxDiv -= 1

# find unique paths .. factorial
m = int(input())
n = int(input())

count_paths = 1

for i in range(1, m):
   count_paths = count_paths*(n-1+i)/i
print(count_paths)

#validate sudoku board
def validate_sudoku_board(board):
   N = 9
   # Use an array to record the status
   rows = [[0] * N for _ in range(N)]
   cols = [[0] * N for _ in range(N)]
   boxes = [[0] * N for _ in range(N)]
   for r in range(N):
         for c in range(N):
            # Check if the position is filled with number
            if board[r][c] == ".":
               continue
            pos = int(board[r][c]) - 1
            # Check the row
            if rows[r][pos] == 1:
                  return False
            rows[r][pos] = 1

            # Check the column
            if cols[c][pos] == 1:
               return False
            cols[c][pos] = 1

         # Check the box
            idx = (r // 3) * 3 + c // 3
            if boxes[idx][pos] == 1:
               return False
            boxes[idx][pos] = 1
   return True

# int palindrome
def isPalindrome(self, x: int) -> bool:
   if x < 0 or (x % 10 == 0 and x != 0):
      return False

   revertedNumber = 0
   while x > revertedNumber:
      revertedNumber = revertedNumber * 10 + x % 10
      x //= 10
   return x == revertedNumber or x == revertedNumber // 10

# palindrome asa string
int_input = input()
flag = True
if int_input[0] == '-':
   flag = False
else:
   for i in range(len(int_input)//2):
      if int_input[i] != int_input[len(int_input)-1-i]:
         flag = False
         break
print(flag)

# numeric palindrome, not converting to string 
def isPalindrome(x):
   if x < 0:
	   return False
   for i in range(len(str(x))//2):
      first = x // 10 ** (len(str(x))-1-i) % 10
      last =  x // (10 ** i) % 10
      if first != last:
         return False
   return True

# wave array
def wave_array(integers):
   integers.sort()
   for i in range(0,len(integers)-1,2):
      integers[i],integers[i+1] = integers[i+1],integers[i]
   return integers

# rotate array by k
def rotate_array(nums, k):
   n = len(nums)
   k %= n

   start = count = 0
   while count < n:
      current, prev = start, nums[start]
      while True:
         next_idx = (current + k) % n
         nums[next_idx], prev = prev, nums[next_idx]
         current = next_idx
         count += 1

         if start == current:
               break
      start += 1

# power of 3
def powerOfThree(n):
   if n == 0:
      return False
   elif n == 1:
      return True
   elif n % 3 == 0:
      return powerOfThree(n//3)
   else:
	   return False
	      
n = int(input())
print(powerOfThree(n))

#string to int
def myAtoi(self, input: str) -> int:
        sign = 1
        result = 0
        index = 0
        n = len(input)

        INT_MAX = pow(2, 31) - 1
        INT_MIN = -pow(2, 31)

        # Discard all spaces from the beginning of the input string.
        while index < n and input[index] == " ":
            index += 1

        # sign = +1, if it's positive number, otherwise sign = -1.
        if index < n and input[index] == "+":
            sign = 1
            index += 1
        elif index < n and input[index] == "-":
            sign = -1
            index += 1

        # Traverse next digits of input and stop if it is not a digit.
        # End of string is also non-digit character.
        while index < n and input[index].isdigit():
            digit = int(input[index])

            # Check overflow and underflow conditions.
            if (result > INT_MAX // 10) or (
                result == INT_MAX // 10 and digit > INT_MAX % 10
            ):
                # If integer overflowed return 2^31-1, otherwise if underflowed return -2^31.
                return INT_MAX if sign == 1 else INT_MIN

            # Append current digit to the result.
            result = 10 * result + digit
            index += 1

        # We have formed a valid number without any overflow/underflow.
        # Return it after multiplying it with its sign.
        return sign * result

# repeated character
def firstRepeatedCharacter(str1):
   unique_chars = []
 
   if len(str1) > 0:
       for char in str1:
          if char not in unique_chars:
             unique_chars.append(char)
          else: 
           return char.lower()
   return ""
     
# first unique char
def firstUniqChar(self, s: str) -> int:
        count_chars = {}

        for char in s:
            if char in count_chars:
                count_chars[char] += 1
            else:
                count_chars[char] = 1
        
        for index, char in enumerate(s):
            if count_chars[char] == 1:
                return index
        return -1
# repeated substring , brute force
def repeatedSubstring(s):
   uniq = []
   for char in s:
      if char not in uniq:
         uniq.append(char)
      else:
         break
   substr = ''.join(uniq)
   if len(s) % len(substr) == 0:
      for i in range(0,len(s)//2, len(substr)):
         for j in range(len(substr)):
            if s[i+j]  != substr[j]:
               return False
   else:
      return False
            
   return True

#target sum of 2 elements of sorted array
def check_for_target(nums, target):
   left = 0
   right = len(nums)-1

   while left < right:
      sum = nums[left]+nums[right]
      if sum == target:
         return True
      elif sum > target:
         right -= 1
      else:
         left += 1
   return False
      
# zip 2 sorted arrays
def combine(arr1, arr2):
   i = j = 0
   ans = []
   while i < len(arr1) and j < len(arr2):
      if arr1[i] < arr2[j]:
         ans.append(arr1[i])
         i += 1
      else:
         ans.append(arr2[j])
         j += 1
   while i < len(arr1):
      ans.append(arr1[i])
      i += 1
   while j < len(arr2):
      ans.append(arr2[j])
      j += 1
   return ans

#reverse string 
def reverse(strlist):
   j = len(strlist)-1

   for i in range(len(strlist)//2):
      strlist[i], strlist[j] = strlist[j],strlist[i]
      j -= 1

   return strlist

# sorted squares 
   def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums)-1
        result = [0]*(len(nums))
        
        for i in range(len(result)-1,-1,-1):
            if abs(nums[left]) > abs(nums[right]):
                result[i] = nums[left]**2
                left += 1
            else:
                result[i] = nums[right]**2
                right -= 1
        return result

# longest valid subarray (sum elements = k)
def valid_sum_sub(nums, k):
   left=curr=answer=0
   for right in range(len(nums)):
      curr += nums[right]
      while curr > k:
         curr -= nums[left]
         left += 1
      answer = max(answer, right - left + 1)
   return answer

#longest substring with 1 zero
def find_length(s):
   left=curr=answer=0
   for right in range(len(s)):
      if s[right] == "0":
         curr += 1
      while curr > 1:
         if s[left] == "0":
            curr -= 1
         left += 1
      answer = max(answer, right-left +1)
   return answer


def numSubarrayProductLessThanK(nums, k):

   if k <= 1:
      return 0
   
   left=answer = 0
   curr = 1
   for right in range(len(nums)):
      curr *= nums[right]

      while curr >= k:
         curr //= nums[left]
         left += 1
      answer += right-left+1
   return answer

def find_best_subarray(nums, k):
   curr = 0
   for i in range(k):
      curr += nums[i]
   ans = curr
   for i in range(k, len(nums)):
      curr += nums[i] - nums[i-k]
      ans = max(ans, curr)
   return ans

def findMaxAverage(nums, k):
   curr=ans=0
   for i in range(k):
      curr += nums[i]
   ans = curr/k
   for i in range(k, len(nums)):
      curr += nums[i] - nums[i-k]
      ans = max(ans, curr/k)
   return ans
#
def longestOnes(nums,k):
   ans=curr=left = 0
   for right in range(len(nums)):
      if nums[right] == 0:
         curr += 1
      while curr > k:
         if nums[left] == 0:
            curr -= 1
         left += 1
      ans = max(ans, right-left+1)
   return ans