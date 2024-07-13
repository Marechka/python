
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


