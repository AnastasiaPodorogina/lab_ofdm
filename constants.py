import bitarray

BPSK = {'-1.0+0.0j':bitarray.bitarray('0'), '-1.0+0.0j':bitarray.bitarray('1')}

QPSK = {'-1.0-1.0j':bitarray.bitarray('00'), '-1.0+1.0j':bitarray.bitarray('01'), 
		'1.0-1.0j':bitarray.bitarray('10'), '1.0+1.0j':bitarray.bitarray('11')}

QAM = {'-3.0+3.0j':bitarray.bitarray('0000'), '-3.0+1.0j':bitarray.bitarray('0001'), 
		  '-1.0+3.0j':bitarray.bitarray('0100'), '-1.0+1.0j':bitarray.bitarray('0101'), 
		  '1.0+3.0j':bitarray.bitarray('1100'), '3.0+3.0j':bitarray.bitarray('1000'), 
	  	  '1.0+1.0j':bitarray.bitarray('1101'), '3.0+1.0j':bitarray.bitarray('1001'), 
 	  	  '-3.0-1.0j':bitarray.bitarray('0011'), '-1.0-1.0j':bitarray.bitarray('0111'),
	  	  '-3.0-3.0j':bitarray.bitarray('0010'), '-1.0-3.0j':bitarray.bitarray('0110'), 
	  	  '1.0-1.0j':bitarray.bitarray('1111'), '3.0-1.0j':bitarray.bitarray('1011'), 
	  	  '1.0-3.0j':bitarray.bitarray('1110'), '3.0-3.0j':bitarray.bitarray('1010')}

N_c = 200
N_fft = 1024
BPSK_data_arr = [-1, 1]

QPSK_data_arr = [-1.-1.j, -1.+1.j, 1.-1.j, 1.+1.j]

QAM_data_arr = [-3.+3.j, -3.+1.j, -1.+3.j, -1.+1.j, 1.+3.j, 3.+3.j, 1.+1.j, 3.+1.j, 
			 	-3.-1.j, -1.-1.j, -3.-3.j, -1.-3.j, 1.-1.j, 3.-1.j, 1.-3.j, 3.-3.j]