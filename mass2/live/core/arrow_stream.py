"""Write, and follow while it grows, an Arrow IPC stream file of polars DataFrames.

Arrow IPC *stream* files (".arrows"), not the IPC *file* format (Feather v2): the file format writes its schema
and batch index in a footer only on close, so nothing can read it while it grows. A stream is a sequence of
self-delimiting messages (schema, batch, batch, ..., end-of-stream marker) that can be read as it is appended.
polars writes and reads complete streams but cannot append to one or read one incrementally; these two do that
with pyarrow, and otherwise deal only in polars DataFrames.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import polars as pl
import pyarrow as pa

# Every IPC message starts with this 4-byte continuation marker followed by an int32 metadata length.
# A metadata length of zero is the end-of-stream marker.
_CONTINUATION = b"\xff\xff\xff\xff"
_END_OF_STREAM = _CONTINUATION + b"\x00\x00\x00\x00"


@dataclass(frozen=True)
class ArrowStreamTailer:
    """How far an Arrow IPC stream file has been read. `poll(max_bytes)` returns a new tailer, the complete record
    batches appended since (as polars DataFrames, read one message at a time until `max_bytes` have been read, always
    at least one batch if there is one, so memory is bounded by `max_bytes`), and `caught_up`: True when it read every
    complete batch there was, False when it stopped at `max_bytes` with more waiting.

    The file need not exist yet, and its last message may be only partly written: an incomplete message is read
    again on the next poll. `ended` is True once the end-of-stream marker has been read.
    """

    path: Path
    bytes_read: int = 0  # bytes of complete messages consumed
    ended: bool = False
    schema: pa.Schema | None = None

    def poll(self, max_bytes: int) -> tuple["ArrowStreamTailer", list[pl.DataFrame], bool]:
        if self.ended or not self.path.exists():
            return self, [], True
        frames: list[pl.DataFrame] = []
        schema, ended, offset, caught_up = self.schema, False, self.bytes_read, True
        with pa.OSFile(str(self.path)) as f:
            f.seek(offset)
            while True:
                if len(frames) > 0 and offset - self.bytes_read >= max_bytes:
                    caught_up = False
                    break
                if f.read(8) == _END_OF_STREAM:
                    ended, offset = True, offset + 8
                    break
                f.seek(offset)
                try:
                    message = pa.ipc.read_message(f)
                except (pa.ArrowInvalid, OSError, EOFError):
                    break  # nothing more yet, or the writer has not finished this message
                offset = f.tell()
                if message.type == "schema":
                    schema = pa.ipc.read_schema(message)
                elif message.type == "record batch":
                    frames.append(pl.DataFrame(pl.from_arrow(pa.ipc.read_record_batch(message, schema))))
                else:
                    raise ValueError(f"Unsupported Arrow IPC message type {message.type!r} in {self.path}")
        return replace(self, bytes_read=offset, ended=ended, schema=schema), frames, caught_up


@dataclass
class ArrowStreamWriter:
    """Append polars DataFrames, one record batch each, to a new Arrow IPC stream file (an open file, so not frozen).
    The file must not exist yet: an existing file raises FileExistsError rather than being overwritten.

    The first DataFrame fixes the schema. Later frames are conformed to it: columns are reordered and cast,
    missing columns are filled with nulls, and unexpected columns are dropped. Every batch is flushed to the
    OS immediately, so a concurrent `ArrowStreamTailer` sees it on its next poll.
    """

    path: Path
    schema: pl.Schema | None = None
    _file: BinaryIO = field(init=False, repr=False)
    _writer: pa.ipc.RecordBatchStreamWriter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "xb")  # an existing file is an error (FileExistsError), never overwritten

    def write(self, df: pl.DataFrame) -> None:
        """Append `df` as exactly one record batch."""
        if self.schema is None:
            self.schema = df.schema
        else:
            df = conform_to_schema(df, self.schema)
        table = df.to_arrow().combine_chunks()
        if self._writer is None:
            self._writer = pa.ipc.new_stream(self._file, table.schema)
        batch = table.to_batches()[0] if table.num_rows > 0 else pa.RecordBatch.from_pylist([], schema=table.schema)
        self._writer.write_batch(batch)
        self._file.flush()

    def close(self) -> None:
        """Write the end-of-stream marker and close the file. Safe to call twice."""
        if self._file.closed:
            return
        if self._writer is not None:
            self._writer.close()  # writes the end-of-stream marker
        else:
            self._file.write(_END_OF_STREAM)  # an empty, but valid and finished, stream
        self._file.close()

    def __enter__(self) -> "ArrowStreamWriter":
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


def conform_to_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Return `df` with exactly the columns of `schema`, in order, cast to its dtypes; missing columns are null."""
    return df.select([
        pl.col(name).cast(dtype) if name in df.columns else pl.lit(None, dtype).alias(name) for name, dtype in schema.items()
    ])


def write_stream_atomically(df: pl.DataFrame, path: str | Path) -> None:
    """Write `df` as a complete IPC stream, replacing `path` atomically so readers never see a partial file. It is
    written by `ArrowStreamWriter`, so its columns have the same Arrow types as an appended stream's."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)  # left by a process that stopped mid-write
    with ArrowStreamWriter(tmp) as writer:
        writer.write(df)
    os.replace(tmp, path)
