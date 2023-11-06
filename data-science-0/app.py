print("Hello World")


def kuadrat(array):
    temp = []
    for i in range(0, len(array)):
        temp.append(array[i]**2)
    return temp


print(kuadrat([2, 3, 5]))
print(list(map(lambda x: x**2, [2, 3, 5])))