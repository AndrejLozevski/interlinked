import interlinked as lnk
import numpy as np

def test_pearson():
    x = np.zeros(20, np.float64)
    y = np.zeros(20, np.float64)
    x[:] = [i   for i in range(1,21)]
    y[:] = [i*2 for i in range(1,21)]
    result = lnk.stats.pearson(x, y)
    assert result == (1.0, 0.0)

    x = np.zeros(20, np.float64)
    y = np.zeros(20, np.float64)
    x[:] = [i   for i in range(1,21)]
    y[:] = [i*2 for i in range(1,21)][::-1]
    result = lnk.stats.pearson(x, y)
    assert result == (-1.0, 0.0)

    x = np.zeros(20, np.float64)
    y = np.zeros(20, np.float64)
    x[:] = [i   for i in range(1,21)]
    y[:] = [i*2 for i in range(1,21)]
    y[5] = 100
    result = lnk.stats.pearson(x, y)
    assert (0.0 < result[0] < 1.0) and (0.0 < result[1] < 1.0)

