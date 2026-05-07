# Classical compression floors

These are the reference bpb numbers we are trying to beat with the
NNCP-style transformer + arithmetic coder on the prepared `data.bin`
streams in `data/processed*/monaco_pmtiles/`.

## monaco (data/processed/monaco_pmtiles/data.bin, 2,804,201 bytes)

| Method | Compressed bytes | bpb |
| --- | ---: | ---: |
| Uniform (no compression) | 2,804,201 | 8.000 |
| Unigram entropy (Huffman, 0-order) | — | 6.456 |
| gzip -9 | 1,891,534 | 5.396 |
| bzip2 -9 | 1,791,266 | 5.110 |
| zstd -22 --ultra | 1,723,492 | 4.917 |
| **xz -9 (LZMA)** | **1,617,364** | **4.614** |

## paris (data/processed_paris/monaco_pmtiles/data.bin, 51,797,434 bytes)

| Method | Compressed bytes | bpb |
| --- | ---: | ---: |
| gzip -9 | 29,958,996 | 4.627 |
| bzip2 -9 | 28,252,983 | 4.364 |
| zstd -22 --ultra | 26,412,901 | 4.079 |
| **xz -9 (LZMA)** | **24,280,532** | **3.750** |

Paris compresses about 0.86 bpb better than Monaco under xz — there is
more cross-tile redundancy when 434 tiles are concatenated than 37.

## england (data/processed_england/monaco_pmtiles/data.bin, 3,285,573,784 bytes)

| Method | Compressed bytes | bpb |
| --- | ---: | ---: |
| gzip -9 | 1,987,288,510 | 4.839 |
| bzip2 -9 | 1,864,748,310 | 4.541 |
| zstd -19 | 1,820,005,651 | 4.432 |
| **xz -9 (LZMA)** | **1,670,990,528** | **4.069** |

England's classical floors are looser than Paris's (xz 4.07 vs 3.75)
because the dataset spans heterogeneous landscapes (urban + rural +
coast + uplands) rather than a single dense city — less repeated
cross-tile structure for LZMA to chew on. Still tighter than Monaco's
xz 4.61, since 876 K tiles still amortise the schema strings better
than 37 do.
