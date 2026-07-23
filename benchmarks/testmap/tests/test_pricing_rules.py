from pricing import discount

def test_discount():
    assert discount(100, 0.1) == 90
