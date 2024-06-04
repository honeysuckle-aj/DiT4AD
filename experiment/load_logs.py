import os.path
import random

import numpy as np
import matplotlib.pyplot as plt


def load_log():
    folder = "../dataset/logs"
    act_fn = {"GELU": {"loss": [], "auc": []}, "SiLU": {"loss": [], "auc": []}, "ReLU": {"loss": [], "auc": []},
              "ELU": {"loss": [], "auc": []}}
    for act in act_fn.keys():
        log_fn = os.path.join(folder, act, "log.txt")
        auc_fn = os.path.join(folder, act, "auc.txt")
        loss = np.loadtxt(log_fn)
        auc = np.loadtxt(auc_fn)
        act_fn[act]["loss"] = loss
        act_fn[act]["auc"] = auc
    return act_fn


def draw_loss(logs):
    x = np.linspace(0, len(logs["GELU"]["loss"]) * 100, len(logs["GELU"]["loss"]))
    plt.plot(x, np.log(logs["GELU"]["loss"]), c="red", label="GELU")
    plt.plot(x, np.log(logs["SiLU"]["loss"]), c="purple", label="SiLU")
    plt.plot(x, np.log(logs["ReLU"]["loss"]), c="blue", label="ReLU")
    plt.plot(x, np.log(logs["ELU"]["loss"]), c="orange", label="ELU")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.clf()

def draw_auc(logs):
    x = np.linspace(0, len(logs["GELU"]["auc"]) * 100, len(logs["GELU"]["auc"]))
    plt.plot(x, logs["GELU"]["auc"], c="red", label="GELU")
    plt.plot(x, logs["SiLU"]["auc"], c="purple", label="SiLU")
    plt.plot(x, logs["ReLU"]["auc"], c="blue", label="ReLU")
    plt.plot(x, logs["ELU"]["auc"], c="orange", label="ELU")
    plt.xlabel("epochs")
    plt.ylabel("AUC(pixel level)")
    plt.legend()
    plt.savefig("auc.png")
    plt.clf()

if __name__ == "__main__":
    act_logs = load_log()
    draw_loss(act_logs)
    draw_auc(act_logs)
    print("f")
