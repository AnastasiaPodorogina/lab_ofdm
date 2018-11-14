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
	mean_power = np.sqrt(np.mean(power))
	return mean_power


def create_message(path='wind.png'):
	""" Побитовое считывание из файла """
	#generate bitarray
	msg = bitarray.bitarray()
	
	#read message bits from file
	with open(path, 'rb') as file:
		msg.fromfile(file)
	
	return msg


def mapper(msg, dictionary, norm=None):	
	""" Отображение битовой последовательности на созвездие """
	complex_msg = list(map(complex, msg.decode(dictionary)))
	norm_msg = list(map(lambda x,y=norm: x/y, complex_msg)) 
	logging.debug('complex message = {}'.format(complex_msg))
	return complex_msg


# create OFDM-signal

def grouper(N_c, iterable, fillvalue = complex(0)):
	""" Группирует элементы массива по N_carriers """
	groups = list(itertools.zip_longest(*([iter(iterable)]*N_c), fillvalue = fillvalue))
	return np.array(groups)

def add_guard_interval(T_guard, N_fft ,new_group):
	tmp_ifft = copy.deepcopy(new_group)
	tmp = np.insert(tmp_ifft , 0, tmp_ifft[-T_guard:])
	return tmp

def upsampler(N_fft, N_c, groups):
	""" Дополняет массивы до N_fft """
	list_nfft = list()
	
	for group in groups:
		new_group = np.insert(group, 0, np.zeros(1, dtype=complex))
		new2 = np.append(group, np.zeros(N_fft - N_c - 1, dtype=complex))
		list_nfft.append(new2)	
	
	return np.array(list_nfft)	


def OFDM_signal(groups, T_guard, N_fft):
	""" Обратное Фурье преобразование """
	ofdm_frame = list()
	
	for group in groups:
		new_group = np.fft.ifft(group)
		guard_symbol = add_guard_interval(T_guard, N_fft, new_group)
		ofdm_frame.append(guard_symbol)
	
	return ofdm_frame

def remove_guard_interval(T_guard, group):
	tmp = copy.deepcopy(groups)
	return tmp[T_guard - 1:]


def demapper(message, data_arr, dictionary, norm=None):
	""" Определяет к какой точке созвездия относятся принятые биты """
	dist_arr = abs(np.asarray(message).reshape((-1, 1)) - np.asarray(data_arr).reshape((1, -1)))
	min_arg = dist_arr.argmin(axis=1)
	hard_decidion = np.asarray(data_arr)[min_arg]
	
	norm_msg = list(map(lambda x,y=norm: x*y, hard_decidion))	
	bit_msg = bitarray.bitarray()
	ms = list(map(lambda x: '{:.1f}'.format(x), hard_decidion))
	bit_msg.encode(dictionary, ms) 
	return bit_msg[0:66632]
  	  

			  	  
def receiver(N_c, N_fft, ofdm_frame, T_guard):
	""" Фурье преобразование """
	ofdm_simv = list()
	array = np.zeros(N_fft, dtype=complex)
	print(ofdm_frame)
	for group in ofdm_frame:
		array = group
		#arr_1 = remove_guard_interval(T_guard, array)
		arr_1 = array[T_guard:]
		fft_arr = np.fft.fft(arr_1)
		ofdm_simv += fft_arr[: N_c].tolist()
	return ofdm_simv

def scrambler(input_array=None, register=bitarray.bitarray('100101010000000')) -> bitarray:
    """Рандомизатор входной(выходной) последовательности
        :param input_array: входящий поток
        :param register: инициализирующая последовательность
        :return: рандомизированная последовательность
    """
    output_bit_array = bitarray.bitarray()
    temp_input_array = input_array.copy()
    temp_register = register.copy()
    register_size = temp_register.length()

    temp_input_array.reverse()

    for bit in temp_input_array:
        last_bit_xor = temp_register[register_size - 1]
        pre_last_bit_xor = temp_register[register_size - 2]
        xored_bit = pre_last_bit_xor ^ last_bit_xor
        temp_register.insert(0, xored_bit)
        temp_register.pop()
        input_bit_xor = xored_bit ^ bit
        output_bit_array.insert(0, input_bit_xor)

    return output_bit_array	

def peak_factor(msg):
	power = list(map(lambda x: np.square(np.abs(x)), msg))
	max_power = np.max(power)
	mean_power = np.mean(power)
	peak_factor = max_power / mean_power
	return 20*log10(peak_factor)    


def verification(input_msg, output_msg):
	return np.array(input_msg)[0:66632] ^ np.array(output_msg)[0:66632]
	

if __name__ == '__main__':
	N_c = 200
	N_fft = 1024
	T_guard = N_fft // 8

	msg = create_message()
	norm = norm(QAM_data_arr)
	scrambled_msg = scrambler(msg)
	#transmitter
	mapped_msg = mapper(scrambled_msg, QAM, norm)

	ofdm_signal = grouper(N_c, mapped_msg)
	groups = upsampler(N_fft, N_c, ofdm_signal)
	ofdm_frame = OFDM_signal(groups, T_guard, N_fft)
	ofdm_simbol = receiver(N_c, N_fft, ofdm_frame, T_guard)
	print(ofdm_simbol)
	output_msg = demapper(ofdm_simbol, QAM_data_arr, QAM, norm)
	#print(output_msg)
	scrambled_output_msg = scrambler(output_msg)
	#print(msg)
	#print('_____________________')
	#print(scrambled_output_msg)
	verificated = verification(msg, scrambled_output_msg)

	plt.plot(verificated)
	plt.show()




	
	