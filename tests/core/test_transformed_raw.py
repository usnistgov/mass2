import numpy as np
import pulsedata
import mass2


def test_inverted_data():
    """Read the same file normally and inverted. Be sure that all means of
    accssing the raw data give bitwise inverses of each other."""
    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].noise_folder / "20240727_run0000_chan4219.ljh"

    def invert(raw):
        "Perform bitwise inversion of the `raw` array"
        return ~raw

    ds1 = mass2.Channel.from_ljh(src_name)
    ds2 = mass2.Channel.from_ljh(src_name, transform_raw=invert)
    all = range(0, ds1.npulses)
    assert ds1.pulsereader is not None
    assert ds2.pulsereader is not None
    raw1 = ds1.load_raw(all)
    raw2 = ds2.load_raw(all)
    assert ds2.transform_raw is not None
    tr2 = ds2.transform_raw(raw2)

    assert np.all(raw1 == ~raw2)
    assert np.all(raw1 == tr2)
