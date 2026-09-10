# Variadic load report over 12 legs

## Realised prompt-length histogram, read back from each server's own usage.prompt_tokens

| prompt tokens | OURS | THEIRS |
|---|---|---|
| 0-127 | 266 | 266 |
| 128-255 | 260 | 260 |
| 256-511 | 48 | 48 |
| 512-1023 | 96 | 96 |
| 1024-2047 | 26 | 26 |
| 2048-4095 | 72 | 72 |

| band | n | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| `S` | 508 | 78 | 109 | 139 | 160 | 111 |
| `M` | 640 | 93 | 170 | 276 | 457 | 185 |
| `L` | 244 | 739 | 930 | 1127 | 1139 | 933 |
| `XL` | 144 | 2288 | 2962 | 3127 | 3153 | 2811 |

## Per leg: throughput and the spread between rounds

| arm | c | round | ok/n | out tok/s | decode-only tok/s | mean TTFT ms | p95 TTFT ms | wall s | counted by | publishable |
|---|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 128/128 | 35.85 | 59.05 | 1975.1 | 9471.7 | 649.2 | usage | yes |
| OURS | 1 | 2 | 128/128 | 35.77 | 59.02 | 1984.7 | 9474.5 | 650.6 | usage | yes |
| OURS | 4 | 1 | 128/128 | 53.18 | 16.58 | 2701.4 | 10273.8 | 438.8 | usage | yes |
| OURS | 4 | 2 | 128/128 | 53.16 | 16.83 | 2742.8 | 10543.1 | 438.8 | usage | yes |
| OURS | 8 | 1 | 128/128 | 53.83 | 7.86 | 3797.2 | 10837.9 | 432.2 | usage | yes |
| OURS | 8 | 2 | 128/128 | 54.03 | 7.92 | 3910.4 | 11346.3 | 431.4 | usage | yes |
| THEIRS | 1 | 1 | 128/128 | 33.10 | 44.96 | 1359.7 | 4641.4 | 673.8 | usage | yes |
| THEIRS | 1 | 2 | 128/128 | 33.14 | 44.98 | 1350.8 | 4534.6 | 670.8 | usage | yes |
| THEIRS | 4 | 1 | 128/128 | 33.33 | 44.31 | 16681.3 | 21807.4 | 668.2 | usage | yes |
| THEIRS | 4 | 2 | 128/128 | 32.77 | 43.11 | 16972.2 | 23693.9 | 680.5 | usage | yes |
| THEIRS | 8 | 1 | 128/128 | 33.52 | 44.31 | 36600.6 | 45418.1 | 665.5 | usage | yes |
| THEIRS | 8 | 2 | 128/128 | 32.17 | 43.53 | 38348.5 | 48363.8 | 694.9 | usage | yes |

## Round-to-round spread per cell

| arm | c | out tok/s per round | spread | p95 TTFT ms per round | spread |
|---|---|---|---|---|---|
| OURS | 1 | 35.85 / 35.77 | 0.2% | 9471.7 / 9474.5 | 0.0% |
| OURS | 4 | 53.18 / 53.16 | 0.0% | 10273.8 / 10543.1 | 2.6% |
| OURS | 8 | 53.83 / 54.03 | 0.4% | 10837.9 / 11346.3 | 4.7% |
| THEIRS | 1 | 33.10 / 33.14 | 0.1% | 4641.4 / 4534.6 | 2.4% |
| THEIRS | 4 | 33.33 / 32.77 | 1.7% | 21807.4 / 23693.9 | 8.7% |
| THEIRS | 8 | 33.52 / 32.17 | 4.2% | 45418.1 / 48363.8 | 6.5% |

## Percentiles, pooled over both rounds (warm only, warmup discarded)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 256 | ttft ms | 802.2 | 3937.7 | 9798.6 | 10530.2 | 10715.3 | 1979.9 |
| OURS | 1 | 256 | ttft_corrected (est) ms | 802.2 | 3937.7 | 9798.6 | 10530.2 | 10715.3 | 1979.9 |
| OURS | 1 | 256 | tpot ms | 16.1 | 21.8 | 24.3 | 28.3 | 29.5 | 16.9 |
| OURS | 1 | 9781 | itl ms | 80.0 | 86.2 | 87.2 | 88.3 | 119.0 | 81.0 |
| OURS | 1 | 256 | e2el ms | 3944.3 | 8784.7 | 13097.9 | 14608.5 | 15116.0 | 5077.0 |
| OURS | 4 | 256 | ttft ms | 1226.6 | 8980.2 | 10434.1 | 13325.7 | 13523.3 | 2722.1 |
| OURS | 4 | 256 | ttft_corrected (est) ms | 1226.6 | 8980.2 | 10434.1 | 13325.7 | 13523.3 | 2722.1 |
| OURS | 4 | 256 | tpot ms | 50.6 | 97.7 | 113.5 | 159.9 | 175.4 | 59.9 |
| OURS | 4 | 9714 | itl ms | 170.2 | 174.9 | 523.5 | 3364.8 | 10550.5 | 288.5 |
| OURS | 4 | 256 | e2el ms | 11696.2 | 22845.9 | 23954.9 | 31680.3 | 36557.5 | 13670.3 |
| OURS | 8 | 256 | ttft ms | 1921.7 | 9530.0 | 11179.7 | 13711.9 | 13816.3 | 3853.8 |
| OURS | 8 | 256 | ttft_corrected (est) ms | 1921.7 | 9530.0 | 11179.7 | 13711.9 | 13816.3 | 3853.8 |
| OURS | 8 | 256 | tpot ms | 123.2 | 193.5 | 205.3 | 226.1 | 278.2 | 126.7 |
| OURS | 8 | 9711 | itl ms | 328.7 | 731.4 | 1177.3 | 8957.6 | 13258.3 | 605.8 |
| OURS | 8 | 256 | e2el ms | 25732.3 | 40348.0 | 45486.2 | 49377.7 | 54770.1 | 26837.7 |
| THEIRS | 1 | 256 | ttft ms | 736.8 | 3114.9 | 4638.6 | 5476.3 | 5782.2 | 1355.3 |
| THEIRS | 1 | 256 | ttft_corrected (est) ms | 736.8 | 3114.9 | 4638.6 | 5476.3 | 5782.2 | 1352.2 |
| THEIRS | 1 | 256 | tpot ms | 21.3 | 29.9 | 32.0 | 35.1 | 37.9 | 22.2 |
| THEIRS | 1 | 40969 | itl ms | 0.0 | 118.1 | 124.0 | 130.2 | 261.4 | 24.4 |
| THEIRS | 1 | 256 | e2el ms | 4738.2 | 8453.3 | 9573.7 | 10517.8 | 11586.8 | 5252.3 |
| THEIRS | 4 | 256 | ttft ms | 16711.0 | 20614.1 | 22786.9 | 26147.6 | 28039.2 | 16826.7 |
| THEIRS | 4 | 256 | ttft_corrected (est) ms | 16711.0 | 20604.4 | 22783.9 | 26147.6 | 28039.2 | 16823.4 |
| THEIRS | 4 | 256 | tpot ms | 22.3 | 30.6 | 33.3 | 35.7 | 38.2 | 22.9 |
| THEIRS | 4 | 40914 | itl ms | 0.0 | 119.0 | 124.3 | 130.7 | 261.2 | 25.1 |
| THEIRS | 4 | 256 | e2el ms | 20739.8 | 24864.5 | 27414.7 | 31256.9 | 33092.2 | 20842.0 |
| THEIRS | 8 | 256 | ttft ms | 37438.0 | 45158.5 | 47361.5 | 49125.7 | 51709.9 | 37474.5 |
| THEIRS | 8 | 256 | ttft_corrected (est) ms | 37428.9 | 45158.5 | 47349.4 | 49121.2 | 51698.3 | 37471.3 |
| THEIRS | 8 | 256 | tpot ms | 22.3 | 30.3 | 31.9 | 33.5 | 37.2 | 22.8 |
| THEIRS | 8 | 41050 | itl ms | 0.0 | 118.7 | 124.7 | 130.9 | 268.3 | 24.9 |
| THEIRS | 8 | 256 | e2el ms | 41194.2 | 49873.6 | 51273.9 | 53809.0 | 55084.1 | 41474.5 |

## Percentiles, pooled over both rounds (with warmup included)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 264 | ttft ms | 801.7 | 3943.2 | 9803.2 | 10552.3 | 27106.0 | 2058.3 |
| OURS | 1 | 264 | ttft_corrected (est) ms | 801.7 | 3943.2 | 9803.2 | 10552.3 | 27106.0 | 2058.3 |
| OURS | 1 | 264 | tpot ms | 16.1 | 21.9 | 24.5 | 28.3 | 29.5 | 17.0 |
| OURS | 1 | 10109 | itl ms | 80.0 | 86.1 | 87.1 | 88.3 | 119.0 | 81.0 |
| OURS | 1 | 264 | e2el ms | 3944.3 | 8788.4 | 13153.5 | 14783.4 | 30434.5 | 5162.5 |
| OURS | 4 | 264 | ttft ms | 1303.9 | 8767.3 | 10416.7 | 13324.1 | 13523.3 | 2777.1 |
| OURS | 4 | 264 | ttft_corrected (est) ms | 1303.9 | 8767.3 | 10416.7 | 13324.1 | 13523.3 | 2777.1 |
| OURS | 4 | 264 | tpot ms | 49.9 | 97.4 | 113.1 | 157.8 | 175.4 | 59.1 |
| OURS | 4 | 10038 | itl ms | 170.2 | 174.8 | 521.5 | 3303.1 | 10550.5 | 284.1 |
| OURS | 4 | 264 | e2el ms | 11638.6 | 22821.4 | 23946.1 | 31661.1 | 36557.5 | 13582.1 |
| OURS | 8 | 272 | ttft ms | 2074.8 | 10783.1 | 11928.5 | 20681.2 | 20684.2 | 4446.2 |
| OURS | 8 | 272 | ttft_corrected (est) ms | 2074.8 | 10783.1 | 11928.5 | 20681.2 | 20684.2 | 4446.2 |
| OURS | 8 | 272 | tpot ms | 119.6 | 190.7 | 204.8 | 226.1 | 278.2 | 122.9 |
| OURS | 8 | 10340 | itl ms | 328.5 | 723.8 | 1074.5 | 8901.8 | 13258.3 | 586.2 |
| OURS | 8 | 272 | e2el ms | 25732.3 | 39650.7 | 45195.4 | 49349.9 | 54770.1 | 26735.2 |
| THEIRS | 1 | 264 | ttft ms | 737.5 | 3651.1 | 5013.8 | 5666.5 | 7873.7 | 1418.6 |
| THEIRS | 1 | 264 | ttft_corrected (est) ms | 737.5 | 3651.1 | 5013.8 | 5666.5 | 7873.7 | 1415.4 |
| THEIRS | 1 | 264 | tpot ms | 21.4 | 29.9 | 31.8 | 35.1 | 37.9 | 22.3 |
| THEIRS | 1 | 42260 | itl ms | 0.0 | 118.2 | 124.1 | 130.2 | 923.7 | 24.4 |
| THEIRS | 1 | 264 | e2el ms | 4808.9 | 8525.1 | 9710.4 | 11515.4 | 12918.3 | 5325.1 |
| THEIRS | 4 | 264 | ttft ms | 16590.1 | 20489.8 | 22552.4 | 26086.5 | 28039.2 | 16566.1 |
| THEIRS | 4 | 264 | ttft_corrected (est) ms | 16587.9 | 20486.9 | 22552.4 | 26086.5 | 28039.2 | 16562.7 |
| THEIRS | 4 | 264 | tpot ms | 22.3 | 30.5 | 33.2 | 35.7 | 38.2 | 22.9 |
| THEIRS | 4 | 42203 | itl ms | 0.0 | 119.0 | 124.4 | 130.8 | 261.2 | 25.1 |
| THEIRS | 4 | 264 | e2el ms | 20634.7 | 24815.6 | 27379.5 | 31241.4 | 33092.2 | 20576.0 |
| THEIRS | 8 | 272 | ttft ms | 37265.3 | 45088.8 | 47368.6 | 49242.0 | 51709.9 | 36724.7 |
| THEIRS | 8 | 272 | ttft_corrected (est) ms | 37265.3 | 45088.4 | 47359.8 | 49242.0 | 51698.3 | 36721.1 |
| THEIRS | 8 | 272 | tpot ms | 22.3 | 30.0 | 31.8 | 33.4 | 37.2 | 22.7 |
| THEIRS | 8 | 43607 | itl ms | 0.0 | 118.6 | 124.7 | 130.9 | 268.3 | 24.9 |
| THEIRS | 8 | 272 | e2el ms | 41049.9 | 49601.1 | 51275.1 | 54018.1 | 55084.1 | 40709.4 |

## Streaming granularity, and what it does to TTFT

`first chunk tokens` is an ESTIMATE: streaming carries no per-chunk token count, so it is the first chunk's characters divided by that request's own characters-per-token.

| arm | c | tokens per chunk | chars per token | first chunk tokens (est) | p50 TTFT raw ms | p50 TTFT corrected ms (est) | corrections refused |
|---|---|---|---|---|---|---|---|
| OURS | 1 | 4.87 | 3.60 | 0.58 | 802.2 | 802.2 | 0/256 |
| OURS | 4 | 4.89 | 3.59 | 0.58 | 1226.6 | 1226.6 | 0/256 |
| OURS | 8 | 4.88 | 3.61 | 0.58 | 1921.7 | 1921.7 | 0/256 |
| THEIRS | 1 | 1.09 | 3.58 | 1.02 | 736.8 | 736.8 | 0/256 |
| THEIRS | 4 | 1.09 | 3.58 | 1.03 | 16711.0 | 16711.0 | 0/256 |
| THEIRS | 8 | 1.09 | 3.58 | 1.02 | 37438.0 | 37428.9 | 0/256 |

## Acceptance

| arm | c | round | source | value |
|---|---|---|---|---|
| OURS | 1 | 1 | /metrics | accepted/draft = 18522/34377 = 0.539 |
| OURS | 1 | 2 | /metrics | accepted/draft = 18522/34377 = 0.539 |
| OURS | 4 | 1 | /metrics | accepted/draft = 18628/34055 = 0.547 |
| OURS | 4 | 2 | /metrics | accepted/draft = 18635/33964 = 0.549 |
| OURS | 8 | 1 | /metrics | accepted/draft = 18562/33957 = 0.547 |
| OURS | 8 | 2 | /metrics | accepted/draft = 18599/34048 = 0.546 |
| THEIRS | 1 | 1 | usage | accepted 17346, 0.778 per output token |
| THEIRS | 1 | 2 | usage | accepted 17303, 0.778 per output token |
| THEIRS | 4 | 1 | usage | accepted 17250, 0.774 per output token |
| THEIRS | 4 | 2 | usage | accepted 17253, 0.774 per output token |
| THEIRS | 8 | 1 | usage | accepted 17288, 0.775 per output token |
| THEIRS | 8 | 2 | usage | accepted 17345, 0.776 per output token |

## Cold start: what the first request costs

| arm | c | round | warmup TTFT ms, first request | warm p50 TTFT ms | mean TTFT ms warm | mean TTFT ms with warmup |
|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 27106.0 | 801.7 | 1975.1 | 2153.6 |
| OURS | 1 | 2 | 791.2 | 803.1 | 1984.7 | 1963.0 |
| OURS | 4 | 1 | 4572.3 | 1266.9 | 2701.4 | 2756.8 |
| OURS | 4 | 2 | 4421.9 | 1212.2 | 2742.8 | 2797.5 |
| OURS | 8 | 1 | 7252.7 | 1935.1 | 3797.2 | 3997.9 |
| OURS | 8 | 2 | 20684.2 | 1921.7 | 3910.4 | 4894.5 |
| THEIRS | 1 | 1 | 7839.8 | 738.6 | 1359.7 | 1422.7 |
| THEIRS | 1 | 2 | 7873.7 | 734.9 | 1350.8 | 1414.4 |
| THEIRS | 4 | 1 | 973.1 | 16583.0 | 16681.3 | 16415.7 |
| THEIRS | 4 | 2 | 11089.3 | 17059.2 | 16972.2 | 16716.6 |
| THEIRS | 8 | 1 | 25726.4 | 36498.1 | 36600.6 | 35387.9 |
| THEIRS | 8 | 2 | 31463.0 | 38249.8 | 38348.5 | 38061.5 |
