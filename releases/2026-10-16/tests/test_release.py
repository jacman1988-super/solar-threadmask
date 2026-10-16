from verify_release import verify
def test_release_inventory_and_hashes():
    assert verify()["integrity_verified"]
