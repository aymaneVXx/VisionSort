from visionsort.runtime.demo_assets import ensure_demo_assets


def test_demo_assets_exist():
    assets = ensure_demo_assets()
    assert {"C1", "C2", "C3"} == set(assets)
    for path in assets.values():
        assert path.endswith(".mp4")
