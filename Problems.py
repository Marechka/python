
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