import moss
import pulsedata 



def test_to_df():
    p = pulsedata.pulse_noise_ljh_pairs["20230626"]
    ljh_noise = moss.LJHFile(p.noise_folder/"20230626_run0000_chan4102.ljh")
    df_noise, header_df_noise = ljh_noise.to_polars()
    ljh = moss.LJHFile(p.pulse_folder/"20230626_run0001_chan4102.ljh")
    df, header_df = ljh.to_polars()
