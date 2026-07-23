from orders import total

def test_total():
    assert total([{'price': 2}]) == 2
