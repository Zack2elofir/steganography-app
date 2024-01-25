import numpy as np
import cv2
from numpy import asarray
from PIL import Image, ImageTk
import os
import bitstring
import pywt
import struct
import math
import io
import pyexiv2
from scipy.fftpack import dct, idct
import base64
import copy



dic =   {
        0:'00',
        1:'01',
        2:'10',
        3:'11',
        (0,0):['0','0'],
        (0,1):['10','0'],
        (0,2):['1','0'],
        (0,3):['1'],
        (1,0):['00','0'],
        (1,1):['0','1'],
        (1,2):['10','1'],
        (1,3):['1','1'],
        (2,0):['00'],
        (2,1):['00','1'],
        (2,2):['10'],
        (2,3):['11','0'],
        (3,0):['0'],
        (3,1):['01'],
        (3,2):['01','0'],
        (3,3):['11']
        }

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def binaryToDecimal_2bit(binary):    
    binary1 = ''.join([str(i) for i in binary])
    l = []
    k = []
    for i in range(len(binary1)//2):
        l.append(int(binary1[i*2:(i+1)*2],2))
    for j in range(len(l)//2):
        k.append((abs(l[j*2]-l[j*2+1]),(l[j*2],l[j*2+1])))
    return k

def float_2_bin(ini_string):
    n = int(ini_string, 16)  
    bStr = '' 
    while n > 0: 
        bStr = str(n % 2) + bStr 
        n = n >> 1    
    return bStr

def dec_2_bin(ini_string):
    n = int(ini_string, 10)  
    bStr = '' 
    while n > 0: 
        bStr = str(n % 2) + bStr 
        n = n >> 1    
    return '00'[len(bStr):] + bStr

def pixel_2_bin(value):
    f1 = bitstring.BitArray(float=value, length=32)
    return f1.bin

def bin_2_pixel(s):
    f = int(str(s), 2)
    return struct.unpack('f', struct.pack('I', f))[0]


def dwtenc(img, mes, out_f):
    img = cv2.imread(img)
    mes_bits = tobits(mes)
    pr=binaryToDecimal_2bit(mes_bits)
    pr_list = []
    for k in pr:
        if k[1] in dic:
            pr_list.append((dic[k[0]],dic[k[1]]))
    r,g,b = cv2.split(img)
    r = np.array(r,dtype=np.float64)
    g = np.array(g,dtype=np.float64)
    b= np.array(b,dtype=np.float64)
    coeffs1 = pywt.dwt2(r, 'haar')
    co = np.array(coeffs1[0],dtype=np.float64)
    lh = coeffs1[1][0]
    hl = coeffs1[1][1]
    hh = coeffs1[1][2]
    p,q = np.shape(lh)
    for i,j in enumerate(pr_list):
        row = math.floor(i/q)
        col = round(q*(i/q-math.floor(i/q)))
        pix_co = pixel_2_bin(co[row,col])
        pix1_co = pix_co[:-len(j[0])]+j[0]
        co[row,col] = bin_2_pixel(pix1_co)
        pix_lh = pixel_2_bin(lh[row,col])
        pix1_lh = pix_lh[:-len(j[1][0])]+j[1][0]
        lh[row,col] = bin_2_pixel(pix1_lh)
        pix_hl = pixel_2_bin(hl[row,col])
        if len(j[1])==2:
            pix1_hl = pix_hl[:-len(j[1][1])]+j[1][1]
        hl[row,col] = bin_2_pixel(pix1_hl)
    lh = np.array(lh,dtype=np.float64)
    hl = np.array(hl,dtype=np.float64)
    hh = np.array(hh,dtype=np.float64)
    met,new_co = np.modf(co)
    new_r = pywt.idwt2((new_co,(lh,hl,hh)), 'haar')
    w,h = np.shape(new_r)
    g = cv2.resize(g, (h,w))
    b = cv2.resize(b, (h,w))
    sImg = cv2.merge((new_r,g,b))
    cv2.imwrite(out_f,sImg)
    np.save("DWT_image",[met,pr_list],allow_pickle=True)
    return sImg

def dwtdec(img):
    if type(img)==str:
        img = cv2.imread(img)
    meta = np.load("dwt_image.npy",allow_pickle=True)
    r,g,b = cv2.split(img)
    coeffs1 = pywt.dwt2(r, 'haar')
    lh = coeffs1[1][0]
    hl = coeffs1[1][1]
    hh = coeffs1[1][2]
    co = (coeffs1[0]+meta[0])
    kk_list = []
    for i in meta[1]:
        for k,v in dic.items():
            if v==i[1]:
                kk_list.append(k[0])
                kk_list.append(k[1])
    binary = ''.join([dec_2_bin(str(i)) for i in kk_list])
    n = int(binary, 2)
    # print(n.to_bytes((n.bit_length() + 7) // 8, 'big').decode('ascii'))
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode('ascii')


