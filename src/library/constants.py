"""
Module contains reusable constants.
"""
import copy

import astropy.constants

# simulation-specific
HUBBLE = 0.6774
X_H = 0.76

# physical constants (copied to minimize lookup time)
G = copy.copy(astropy.constants.G.cgs.value)
k_B = copy.copy(astropy.constants.k_B.cgs.value)
kpc = copy.copy(astropy.constants.kpc.cgs.value)
m_p = copy.copy(astropy.constants.m_p.cgs.value)
M_sol = copy.copy(astropy.constants.M_sun.cgs.value)

# simulation redshift map
REDSHIFTS = [
    20.046490988807516,
    14.989173240042412,
    11.980213315300293,
    10.975643294137885,
    9.996590466186332,
    9.388771271940549,
    9.00233985416247,
    8.449476294368743,
    8.012172948865935,
    7.5951071498715965,
    7.23627606616736,
    7.005417045544533,
    6.491597745667503,
    6.0107573988449,
    5.846613747881867,
    5.5297658079491026,
    5.227580973127337,
    4.995933468164624,
    4.664517702470927,
    4.428033736605549,
    4.176834914726472,
    4.0079451114652676,
    3.7087742646422353,
    3.4908613692606485,
    3.2830330579565246,
    3.008131071630377,
    2.8957850057274284,
    2.7331426173187188,
    2.5772902716018935,
    2.4442257045541464,
    2.3161107439568918,
    2.207925472383703,
    2.1032696525957713,
    2.0020281392528516,
    1.9040895435327672,
    1.822689252620354,
    1.7435705743308647,
    1.6666695561144653,
    1.6042345220731056,
    1.5312390291576135,
    1.4955121664955557,
    1.4140982203725216,
    1.3575766674029972,
    1.3023784599059653,
    1.2484726142451428,
    1.2062580807810006,
    1.1546027123602154,
    1.1141505637653806,
    1.074457894547674,
    1.035510445664141,
    0.9972942257819404,
    0.9505313515850327,
    0.9230008161779089,
    0.8868969375752482,
    0.8514709006246495,
    0.8167099790118506,
    0.7910682489463392,
    0.7574413726158526,
    0.7326361820223115,
    0.7001063537185233,
    0.6761104112134777,
    0.6446418406845371,
    0.6214287452425136,
    0.5985432881875667,
    0.5759808451078874,
    0.5463921831410221,
    0.524565820433923,
    0.5030475232448832,
    0.4818329434209512,
    0.4609177941806475,
    0.4402978492477432,
    0.41996894199726653,
    0.3999269646135635,
    0.38016786726023866,
    0.36068765726181673,
    0.3478538418581776,
    0.32882972420595435,
    0.31007412012783386,
    0.2977176845174465,
    0.2733533465784399,
    0.2613432561610123,
    0.24354018155467028,
    0.22598838626019768,
    0.21442503551449454,
    0.19728418237600986,
    0.1803852617057493,
    0.1692520332436107,
    0.15274876890238098,
    0.14187620396956202,
    0.12575933241126092,
    0.10986994045882548,
    0.09940180263022191,
    0.08388443079747931,
    0.07366138465643868,
    0.058507322794512984,
    0.04852362998180593,
    0.0337243718735154,
    0.023974428382762536,
    0.009521666967944764,
    0.0,
]
