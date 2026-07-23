from strutil import slugify

def test_slug():
    assert slugify('A B') == 'a-b'
