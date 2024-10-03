#!/bin/bash python

import csv
import numpy as np
import xlrd
from matplotlib import pyplot as plt
import os
import pandas as pd


def exact_data(f):
    col_types = [float]
    f_csv = csv.reader(f)
    data = []
    index = 1
    for col in f_csv:
        col = tuple(convert(value) for convert, value in zip(col_types, col))
        data.append(col[0])
        index += 1

    return np.array(data)


def plot_data(rec_num, name_1="obs_info", name_2=" Dx ", name_3="s_ahead"):
    if name_1 == "control_info":

        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num + '/control_info_record.csv')

            Cur_acc = d1.Cur_acc.values
            q = d1.q.values
            du = d1.du.values

            t = np.arange(0, Cur_acc.shape[0])
            plt.figure()
            plt.plot(t[0:-1], Cur_acc[0:-1], color="blue", label='Cur_acc')
            plt.ylim(-3, 3)

            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], q[10:-1], color="red", label="q")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], du[10:-1], color="blue", label="du")
            plt.legend()

            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_1 + '/control_info_record.csv')
            d2 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_2 + '/control_info_record.csv')

            u1 = eval("d1." + name_2 + ".values")
            u2 = eval("d2." + name_2 + ".values")

            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[10:-1], u1[10:-1], color="blue", label=str(record_num_1) + "_" + name_2)
            plt.plot(t2[10:-1], u2[10:-1], color="red", label=str(record_num_2) + "_" + name_2)
            plt.ylim(-3, 3)
            plt.legend()
            plt.show()

    elif name_1 == "vehicle_info":

        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num + '/vehicle_info_record.csv')

            vs = d1.vs.values
            vs_cmd = d1.vs_cmd.values
            throttle = d1.throttle.values

            t = np.arange(0, vs.shape[0])
            plt.figure()
            plt.plot(t[10:-1], vs[10:-1], color="blue", label='vs')
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], vs_cmd[10:-1], color="red", label="vs_cmd")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], throttle[10:-1], color="red", label="throttle")
            plt.legend()
            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_record.csv')
            d2 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_2 + '/vehicle_info_record.csv')
            u1 = eval("d1." + name_2 + ".values")
            u2 = eval("d2." + name_2 + ".values")

            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[0:-1], u1[0:-1], color="blue", label=str(record_num_1) + "_" + name_2)
            plt.plot(t2[0:-1], u2[0:-1], color="red", label=str(record_num_2) + "_" + name_2)
            plt.legend()
            plt.show()
    else:

        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num + '/obs_info_record.csv')

            ego_s = d1.ego_s.values
            Ds = d1.Ds.values
            vs = d1.vs.values
            Dv = d1.Dv.values
            t = np.arange(0, ego_s.shape[0])
            plt.figure()
            plt.plot(t[10:-1], ego_s[10:-1], color="blue", label='ego_s')
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], Ds[10:-1], color="red", label="Ds")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], vs[10:-1], color="red", label="vs")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], Dv[10:-1], color="red", label="Dv")
            plt.legend()

            plt.show()


        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_1 + '/obs_info_record.csv')
            d2 = pd.read_csv(
                os.path.abspath(
                    '../Closed-loop-self-learning-OS_branch') + "/datas/plot_data_record/" + record_num_2 + '/obs_info_record.csv')

            u1 = eval("d1." + name_2 + ".values")
            u2 = eval("d2." + name_2 + ".values")

            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[0:-1], u1[0:-1], color="blue", label=str(record_num_1) + "_" + name_2)
            plt.plot(t2[0:-1], u2[0:-1], color="red", label=str(record_num_2) + "_" + name_2)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    # rec_num = [1, 2]

    rec_num = 1

    # rec_num = 169
    '''
    u,du,Cur_acc,q,ru,rdu
    '''
    plot_data(rec_num, 'control_info', 'Cur_acc')

    '''
    vs,vs_cmd,throttle
    '''
    # plot_data(rec_num, 'vehicle_info', 'throttle')

    '''
    ego_s, Ds ,vs , Dv
    
    '''

    # plot_data(rec_num, 'obs_info', 'Dv')
