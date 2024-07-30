import numpy as np
import polars as pl
import os
import collections




class LJHFile():
    TOO_LONG_HEADER=100
    def __init__(self, filename, _limit_pulses = None):
        self.filename = filename
        self.__read_header(self.filename)
        self.dtype = np.dtype([('rowcount', np.int64),
                                ('posix_usec', np.int64),
                                ('data', np.uint16, self.nSamples)])
        if _limit_pulses is not None:
            # this is used to demo the process of watching an 
            # ljh file grow over time
            self.nPulses = min(_limit_pulses, self.nPulses)
        self._mmap = np.memmap(self.filename, self.dtype, mode="r",
                               offset=self.header_size, shape=(self.nPulses,))
        self._cache_i = -1
        self._cache_data = None


    def __read_header(self, filename):
        """Read in the text header of an LJH file.

        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples

        Args:
            filename: path to the file to be opened.
        """
        # parse header into a dictionary
        header_dict = collections.OrderedDict()
        with open(filename, "rb") as fp:
            i = 0
            while True:
                i += 1
                line = fp.readline()
                if line.startswith(b"#End of Header"):
                    break
                elif line == b"":
                    raise Exception("reached EOF before #End of Header")
                elif i > self.TOO_LONG_HEADER:
                    raise IOError("header is too long--seems not to contain '#End of Header'\n"
                                    + "in file %s" % filename)
                elif b":" in line:
                    a, b = line.split(b":", 1)  # maxsplits=1, py27 doesnt support keyword
                    a = a.strip()
                    b = b.strip()
                    a = a.decode()
                    b = b.decode()
                    if a in header_dict and a != "Dummy":
                        print("LJHFile.__read_header: repeated header entry {}".format(a))
                    header_dict[a] = b
                else:
                    continue  # ignore lines without ":"
            self.header_size = fp.tell()
            fp.seek(0)
            self.header_str = fp.read(self.header_size)

        # extract required values from header_dict
        # use header_dict.get for default values
        header_dict["Filename"] = filename
        header_dict["Channel"] = int(header_dict["Channel"])
        header_dict["Timebase"] = float(header_dict["Timebase"])
        self.timebase = header_dict["Timebase"]
        header_dict["Total Samples"] = int(header_dict["Total Samples"])
        self.nSamples = header_dict["Total Samples"]
        header_dict["Presamples"] = int(header_dict["Presamples"])
        self.nPresamples = header_dict["Presamples"]
        # column number and row number have entries like "Column number (from 0-0 inclusive)"
        row_number_k = [k for k in header_dict.keys() if k.startswith("Row number")]
        if len(row_number_k) > 0:
            self.row_number = int(header_dict[row_number_k[0]])
        col_number_k = [k for k in header_dict.keys() if k.startswith("Column number")]
        if len(col_number_k) > 0:
            self.row_number = int(header_dict[col_number_k[0]])
        self.client = header_dict.get("Software Version", "UNKNOWN")
        header_dict["Number of Columns"] = int(header_dict.get("Number of columns", -1))
        self.number_of_columns = header_dict["Number of Columns"]
        header_dict["Number of rows"] = int(header_dict.get("Number of rows", -1))
        self.number_of_rows = header_dict["Number of rows"]
        self.timestamp_offset = float(header_dict.get("Timestamp offset (s)", "-1"))
        self.version_str = header_dict['Save File Format Version']
        # if Version(self.version_str.decode()) >= Version("2.2.0"):
        self.pulse_size_bytes = (16 + 2 * self.nSamples) # dont bother with old ljh
        # else:
        #     self.pulse_size_bytes = (6 + 2 * self.nSamples)
        self.binary_size = os.stat(filename).st_size - self.header_size
        self.header_dict = header_dict
        self.nPulses = self.binary_size // self.pulse_size_bytes
        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the "true value" of nPresamples. For now, assume that only
        # DASTARD clients have this figure correct.
        if "DASTARD" not in self.client:
            self.nPresamples += 3

        # Record the sample times in microseconds
        self.sample_usec = (np.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    def get_header_info_as_dict(self):
        return {"number_of_columns": self.number_of_columns,
                "number_of_rows": self.number_of_rows,
                "timebase_s": self.timebase}
    
    def __repr__(self):
        return f"LJHFile {self.filename}"
    
    def read_trace(self, i):
        return self._mmap[i]["data"]
    
    def to_polars(self, keep_posix_usec=False):
        df = pl.DataFrame({"pulse":self._mmap["data"],
                           "posix_usec":self._mmap["posix_usec"],
                           "rowcount":self._mmap["rowcount"]},
                           schema={"pulse":pl.Array(pl.UInt16, self.nSamples),
                                   "posix_usec":pl.UInt64,
                                   "rowcount":pl.UInt64})
        df = df.select(pl.from_epoch("posix_usec", time_unit="us").alias("timestamp")).with_columns(df)
        if not keep_posix_usec:
            df = df.select(pl.exclude("posix_usec"))
        header_df = pl.DataFrame(self.header_dict)
        return df, header_df

