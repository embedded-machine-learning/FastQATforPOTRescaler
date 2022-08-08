def pr(str,a,b):
    print(f"{str} diff: {(a-b).abs().max()}, mean: {(a-b).abs().mean()}")