def is_nan(x):
    y = (x > 0) + (x <= 0)
    return (y == False).any().item()
