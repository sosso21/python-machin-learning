import random


class RandNumber:
    _min: int
    _max: int

    def __init__(self, min, max):
        self._min = min
        self._max = max
        self.number = random.randint(min, max)

    def num(self):
        return self.number

    def compare(self, x):
        if self.number == x:
            return "correct"
        elif self.number <= x:
            return "lower"
        else:
            return "higher"

    def printNumber(self):
        print(f'the number is {self.number}')

    def setDiff(self, x):
        if (self.number < x & x < self._max):
            self._max = x
        if (self.number > x & x > self._min):
            self._min = x

    def guess(self):
        while True:
            try:
                x = int(
                    input(f'Guess number between {self._min} and {self._max} ~# '))
                compare = self.compare(x)
                print(f'{compare}')
                self.setDiff(x)
                if compare == "correct":
                    print(f'c regl the number is {self.number} !')
                    break
            except ValueError:
                print("it's not a number")
