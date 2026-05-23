#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


people = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

bottoms = np.zeros(3)
for i, (fname, color) in enumerate(zip(fruits, colors)):
    plt.bar(people, fruit[i], width=0.5, bottom=bottoms,
            color=color, label=fname)
    bottoms += fruit[i]

plt.ylabel('Quantity of Fruit')
plt.yticks(range(0, 81, 10))
plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
