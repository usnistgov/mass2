from mass2.calibration.import_asd import parseLine


def test_parseline_single_row():
    fields = {"ASD_Levels": ["element", "spectr_charge", "conf", "term", "j_val", "energy", "unc"]}
    levels: dict = {}
    parseLine(levels, fields, "('Ne',10,'2p','2P*','1/2','8240.0','0.1')")
    parseLine(levels, fields, "('Ne',10,'1s','2S','1/2','0','0'),('Ne',10,'2p','2P*','3/2','8247.5','0.1')")
    assert sorted(levels["Ne"][10].keys()) == ["1s 2S J=1/2", "2p 2P* J=1/2", "2p 2P* J=3/2"]
