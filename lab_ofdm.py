import bitarray
import copy
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import scipy as sp

from constants import QAM, QAM_data_arr
from scipy.fftpack import fft, ifft

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

#Create message bits

def norm(data_arr):
	power = list(map(lambda x: np.square(np.abs(x)), data_arr))
	mean_power = np.mean(power)
	return mean_power


def create_message(path='wind.png'):
	""" Побитовое считывание из файла """
	#generate bitarray
	msg = bitarray.bitarray()
	
	#read message bits from file
	with open(path, 'rb') as file:
		msg.fromfile(file)
	
	return msg


def mapper(msg, dictionary):	
	""" Отображение битовой последовательности на созвездие """
	complex_msg = list(map(complex, msg.decode(dictionary)))
	logging.debug('complex message = {}'.format(complex_msg))
	return complex_msg


# create OFDM-signal

def grouper(N_c, iterable, fillvalue = complex(0)):
	""" Группирует элементы массива по N_carriers """
	groups = list(itertools.zip_longest(*([iter(iterable)]*N_c), fillvalue = fillvalue))
	return np.array(groups)

def upsampler(N_fft, N_c, groups):
	""" Дополняет массивы до N_fft """
	list_nfft = list()
	
	for group in groups:
		new_group = np.insert(group, 0, np.zeros(1, dtype=complex))
		new2 = np.append(group, np.zeros(N_fft - N_c - 1, dtype=complex))
		list_nfft.append(new2)	
	
	return np.array(list_nfft)	


def OFDM_signal(groups):
	""" Обратное Фурье преобразование """
	ofdm_frame = list()
	
	for group in groups:
		new_group = np.fft.ifft(group)
		ofdm_frame.append(new_group)
	
	return ofdm_frame


def demapper(message, data_arr, dictionary):
	""" Определяет к какой точке созвездия относятся принятые биты """
	dist_arr = abs(np.asarray(message).reshape((-1, 1)) - np.asarray(data_arr).reshape((1, -1)))
	min_arg = dist_arr.argmin(axis=1)
	hard_decidion = np.asarray(data_arr)[min_arg]
	
	bit_msg = bitarray.bitarray()
	ms = list(map(lambda x: '{:.1f}'.format(x), hard_decidion))
	bit_msg.encode(dictionary, ms) 
	return bit_msg
  	  

			  	  
def receiver(N_c, N_fft, ofdm_frame):
	""" Фурье преобразование """
	ofdm_simv = list()
	array = np.zeros(N_fft, dtype=complex)
	print(ofdm_frame)
	for group in ofdm_frame:
		array = group
		fft_arr = np.fft.fft(array)
		ofdm_simv += fft_arr[0: N_c].tolist()
	return ofdm_simv


def verification(input_msg, output_msg):
	return np.array(input_msg)[0:66632] ^ np.array(output_msg)[0:66632]
	

if __name__ == '__main__':
	N_c = 200
	N_fft = 1024

	msg = create_message()
	#transmitter
	ofdm_signal = grouper(N_c, mapper(msg, QAM))
	groups = upsampler(N_fft, N_c, ofdm_signal)
	ofdm_frame = OFDM_signal(groups)
	ofdm_simbol = receiver(N_c, N_fft, ofdm_frame)
	output_msg = demapper(ofdm_simbol, QAM_data_arr, QAM)
	verificated = verification(msg, output_msg)

	plt.plot(verificated)
	plt.show()




	
	