def test_private(tmp_path):
    (tmp_path / 'proof').write_text('actual child')
    assert False
