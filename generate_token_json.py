# _*_coding:utf-8_*_

import os
import argparse
import json

from pygments.lexers.ruby import RubyLexer


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def tokenize(program_path):
    lexer = RubyLexer()

    with open(program_path, "r") as f:
        program = f.readlines()

    token_streams = []

    for line in program:
        char = 0  # 文字数
        row = []

        for token_data in lexer.get_tokens(line):
            token = token_data[-1]
            # (トークン文字列, 開始位置, 終了位置)の情報を格納
            row.append((token, char, char + len(token)))
            char += len(token)

        token_streams.append(row)

    return token_streams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=argparse.FileType("w"))
    args = parser.parse_args()

    program_path_list = find_all_files(args.source)
    program_path_list = filter(lambda p: os.path.splitext(p)[1] == ".rb",
                               program_path_list)

    programs_info = {}

    num_of_token = 0
    for program_path in program_path_list:
        programs_info[program_path] = tokenize(program_path)
        num_of_token += len(programs_info[program_path])

    json.dump(programs_info, args.output)
    print num_of_token
