# Linux / macOS / MSYS2.   make            -> release build (-O3, native)
#                          make itt ITT=/path/to/ittapi   -> build with VTune ITT markers
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -g -fno-omit-frame-pointer -Wall -Wextra
LDLIBS  ?= -lopenblas -lm
# macOS (Homebrew): export OPENBLAS=$(brew --prefix openblas); then add -I$OPENBLAS/include -L$OPENBLAS/lib
CFLAGS  += $(EXTRA_CFLAGS)

all: tx gemm_bench stream_triad

tx: tx.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
gemm_bench: gemm_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
stream_triad: stream_triad.c
	$(CC) $(CFLAGS) -o $@ $<
itt: tx.c
	$(CC) $(CFLAGS) -DUSE_ITT -I$(ITT)/include -o tx_itt $< $(ITT)/build/bin/libittnotify.a $(LDLIBS) -ldl -lpthread
check: tx
	./tx check ref && ./tx train ref
clean:
	rm -f tx tx_itt gemm_bench stream_triad
