def fibonacci_sequence(n):
    sequence = []
    a, b = 1, 1
    while len(sequence) < n:
        sequence.append(a)
        a, b = b, a + b
    return sequence

print(fibonacci_sequence(10))
