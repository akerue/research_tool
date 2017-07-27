# _*_coding:utf-8_*_

from fabric.api import run, put, get, cd, task, env, local, settings
from pprint import pprint
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt

env.use_ssh_config = True
env.users = ["y_kobayashi_local"]
env.hosts = ["halley.csg"]


def find_all_files(target_dir):
    for root, dirs, files in os.walk(target_dir):
        yield root
        for f in files:
            yield os.path.join(root, f)


@task
def check_style(project_dir):
    remote_base_dir = "/home/y_kobayashi_local/Workspace/style_checker"

    with cd(remote_base_dir):
        run("./verify_with_context.sh {}".format(project_dir))


@task
def download_imgs(remote_path="/home/y_kobayashi_local/Workspace/style_checker/img"):
    imgs = run("find {} -name '*.png'".format(remote_path)).split("\n")

    for img in imgs:
        img = img.rstrip()
        filename = os.path.basename(img).rstrip()
        local("scp halley.csg:{} ~/Workspace/mlab/research/img/{}".format(img, filename))


@task
def download_jsons(remote_path="/home/y_kobayashi_local/Workspace/style_checker/json"):
    jsons = run("find {} -name '*.json'".format(remote_path)).split("\n")

    for j in jsons:
        j = j.rstrip()
        filename = os.path.basename(j).rstrip()
        local("scp halley.csg:{} ~/Workspace/mlab/research/sc_result/{}".format(j, filename))


@task
def download_scripts(remote_path="/home/y_kobayashi_local/Workspace/style_checker/test"):
    scripts = run("find {} -name '*.rb'".format(remote_path)).split("\n")

    for script in scripts:
        script = script.rstrip()
        filename = os.path.basename(script).rstrip()
        local("scp halley.csg:{} ~/Workspace/mlab/research/test/{}".format(script, filename))


@task
def analyze_with_rubo(project_dir, result_dir="rubo_result", conf_path="rubo_conf/rubocop.yml"):
    for path in find_all_files(project_dir):
        if os.path.splitext(path)[1] == ".rb":
            result_path = os.path.join(result_dir,
                                       os.path.basename(path).replace(".rb", ".json"))
            with settings(warn_only=True):
                local("/Users/Robbykunsan/.rbenv/shims/rubocop {} -c {} --format json --out {}".format(path, conf_path, result_path))


def evaluate_with_rubo(tokens, json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    detected = []

    for r in results["files"][0]["offenses"]:
        # rubocopの結果はカウントの最初が1からなので、1引く
        row_num = r["location"]["line"] - 1
        col_num = r["location"]["column"] - 1
        length = r["location"]["length"]

        for i in range(length):
            point = col_num + i
            token_index = 0
            for t in tokens[row_num]:
                if t[1] <= point and point < t[2]:
                    detected.append((row_num, token_index))
                    break
                token_index += 1

    return list(set(detected))


def evaluate_with_sc(tokens, json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    detected = []

    for p in results["points"]:
        token_index = 0
        for t in tokens[p[0]]:
            if t[1] <= p[1] and p[1] < t[2]:
                detected.append((p[0], token_index))
                break
            token_index += 1

    return list(set(detected))


@task
def calculate(token_dir="./tokens", rubo_dir="./rubo_result", sc_dir="./sc_result"):
    # 評価対象のプログラムパスとそこに含まれるトークンが登録される
    token_dict = {}

    for path in find_all_files(token_dir):
        if os.path.splitext(path)[1] == ".json":
            with open(path, "r") as f:
                token_dict.update(json.load(f))

    rubo_collection = []
    sc_collection = []
    for program_path, tokens in token_dict.items():
        print("[info] {}を解析します".format(program_path))
        program_name = os.path.basename(program_path)
        json_name = program_name.replace(".rb", ".json")

        rubo_flag = False
        sc_flag = False

        if os.path.exists(os.path.join(rubo_dir, json_name)):
            rubo_flag = True

        if os.path.exists(os.path.join(sc_dir, json_name)):
            sc_flag = True

        if rubo_flag and sc_flag:
            try:
                rubo_index = evaluate_with_rubo(tokens, os.path.join(rubo_dir, json_name))
                sc_index = evaluate_with_sc(tokens, os.path.join(sc_dir, json_name))
            except:
                print("なんらかのエラーが発生したので次へ")
                continue

            rubo_mesh = []
            sc_mesh = []

            for i in range(len(tokens)):
                rubo_mesh.append([])
                sc_mesh.append([])
                for j in range(len(tokens[i])):
                    rubo_mesh[i].append(0)
                    sc_mesh[i].append(0)

            # 正解データ(Rubocop)
            for index in rubo_index:
                rubo_mesh[index[0]][index[1]] = 1

            rubo_list = [flatten for inner in rubo_mesh for flatten in inner]

            # 予測データ(Style Checker)
            for index in sc_index:
                sc_mesh[index[0]][index[1]] = 1

            sc_list = [flatten for inner in sc_mesh for flatten in inner]

            print(classification_report(rubo_list, sc_list,
                  target_names=["Correct point", "Wrong point"]))

            rubo_collection.extend(rubo_list)
            sc_collection.extend(sc_list)

            #print("Precision: {0:.3f}, Recall: {1:.3f}, F-value: {2:.3f}".format(
            #    precision_score(rubo_list, sc_list, average="binary"),
            #    recall_score(rubo_list, sc_list, average="binary"),
            #    f1_score(rubo_list, sc_list, average="binary")))

        if rubo_flag and not sc_flag:
            print("Style Checkerの結果がありません")

        if not rubo_flag and sc_flag:
            print("Rubocopの結果がありません")

        print("----------------------------------\n")

    print("総合的な結果")
    print(classification_report(rubo_collection, sc_collection,
          target_names=["Correct point", "Wrong point"]))


@task
def clean_up():
    local("rm -f test/*")
    local("rm -f json/*")
    local("rm -f sc_result/*")
    local("rm -f rubo_result/*")


@task
def evaluate():
    clean_up()
    # check_style("test")
    download_jsons()
    download_scripts()
    local("python generate_token_json.py --source test --output tokens/result.json")
    analyze_with_rubo("test")
    calculate()


@task
def parse_result(result_file="./result.log"):
    with open(result_file, "r") as f:
        results = f.readlines()

    regex = r"([+-]?[0-9]+\.?[0-9]*)"
    data_list = np.empty((0, 4), float)

    for l in results:
        if l.startswith("  Wrong point"):
            data = re.findall(regex, l)
            data = [float(i) for i in data]
            data_list = np.append(data_list, np.array([data]), axis=0)

    print len(data_list)
    return data_list


@task
def draw_plot(result_file="./result.log"):
    data_list = parse_result(result_file)

    precision_list = data_list[:, 0]
    recall_list = data_list[:, 1]

    plt.plot(precision_list, recall_list, ".")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Precision", fontsize=20)
    plt.ylabel("Recall", fontsize=20)
    plt.title("Result of analyzing Cookpad intra source codes")

    plt.show()


@task
def draw_histogram(result_file="./result.log"):
    data_list = parse_result(result_file)

    f_list = data_list[:, 2]

    plt.hist(f_list, bins=100)
    plt.show()

