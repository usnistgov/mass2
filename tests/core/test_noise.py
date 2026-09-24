import pulsedata
from mass2 import NoiseResult
import mass2.core.noise_analysis as NA


def test_noise_analysis(tmp_path):
    src_name = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].noise_folder
    savefile = tmp_path / "noise_analysis.parquet"
    NA.analyze_noise_directory(src_name, savefile=savefile)

    nr = NoiseResult.from_parquet(src_name, 4220)
    assert nr.psd.shape == (251,)
    assert nr.autocorr_vec is not None
    assert nr.autocorr_vec.shape == (500,)
    assert nr.dt == 4e-6
