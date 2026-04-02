import numpy as np
import interlinked as nex

def main():
    x = np.array([i   for i in range(10)])
    y = np.array([i*2 for i in range(10)])
    result = nex.pearson(x, y)
    print(result)

if __name__ == "__main__":
    main()

