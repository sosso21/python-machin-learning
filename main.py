with open("number.txt", "w") as f:
    for n in range(0, 11):
        f.write(f"{n}^2 = {n**2}\n")

with open("number.txt", "r") as f:

    print(f.read())
