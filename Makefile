
SRC = src
BIN = bin
OBJ = $(BIN)/obj
INCLUDE = include
LIB = lib

TARGET = $(BIN)/run

CC = clang
CFLAGS = -Wall -Wextra -Werror -Wpedantic -std=c11 -pedantic -I$(INCLUDE) -glldb
LDFLAGS = -lm

FORMATTER = clang-format
FORMAT_STYLE = {BasedOnStyle: google, IndentWidth: 4, ColumnLimit: 120}
FORMAT_FILES = $(shell find . -iname *.h -o -iname *.c)

SRC_FILES = $(shell find $(SRC) -iname *.c)
OBJ_FILES = $(patsubst $(SRC)%.c, $(OBJ)%.o, $(SRC_FILES))
INCLUDE_FILES = $(shell find $(INCLUDE) -iname *.h)

.PHONY: format all run clean

default: all

$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ)/%.o: $(SRC)/%.c $(INCLUDE_FILES)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

all: $(TARGET)

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BIN)

format:
	clang-format -style="$(FORMAT_STYLE)" -i $(FORMAT_FILES)

