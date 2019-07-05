#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Diana Chajkovska
# Date:     20th May, 2019

import random
import string
import textwrap
import time

import XOR_to_use


BLOCK_SIZE_BYTE = 8
BLOCK_SIZE = 64
NUM_OF_ROUNDS = 16


'''Functions to process plain text'''


# PKCS#7 - N bytes, each of value N are added.
def pkcs7(text):
    if len(text) % BLOCK_SIZE_BYTE == 0:
        return text
    add_text = BLOCK_SIZE_BYTE - len(text) % BLOCK_SIZE_BYTE
    added_text = text + (add_text * chr(add_text))
    return added_text


# Returns padded message to plain text
def unpad(text):
    pattern = text[-1]
    length = ord(pattern)
    # check if the bytes to be removed are all the same pattern
    if text.endswith(pattern * length):
        return text[:-length]
    else:
        return text


# Divides text on blocks
def text_by_blocks(text):
    blocks = []
    while text:
        blocks.append(text[:BLOCK_SIZE])
        text = text[BLOCK_SIZE:]
    return blocks


# Recreate the string from the bit array
def bit_array_to_string(array):
    res = ''.join([chr(int(y, 2)) for y in [''.join([str(x) for x in _bytes]) for _bytes in nsplit(array, 8)]])
    return res


# Split a list into sublists of size "n"
def nsplit(s, n):
    return [s[k:k + n] for k in range(0, len(s), n)]


# Transform string to binary string
def to_bin_str(text):
    bin_str = ''
    for i in text:
        ascii_sym = ord(i)
        # print(ascii_sym)
        bin_str += (bin(ascii_sym)[2:]).zfill(8)
    return bin_str


# First step in Feistel network
# Split permuted message in half
def split_msg_in_half(binarybits):
    return binarybits[:32], binarybits[32:]


# Function to permute block of message by scheme(matrix)
def transpose(block, matrix):
    transpose_block = ''
    for i in matrix:
        transpose_block += block[int(i) - 1]
    return transpose_block


'''Functions to process key'''


# Generate a random string of fixed length as a key.
def random_string(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.sample(letters, string_length))


# Split key in half
def split_key_in_half(k):
    return k[:28], k[28:]


# Left shift on the num_of_bits
def circular_left_shift(half_key, num_of_bits):
    shifted_key = half_key[num_of_bits:] + half_key[:num_of_bits]
    return shifted_key


# Making an array of 16 round keys from primary key
def round_keys(key56):
    round_keys_arr = []
    for i in range(NUM_OF_ROUNDS):
        l_side, r_side = split_key_in_half(key56)
        shifted_left = circular_left_shift(l_side, round_shifts[i])
        shifted_right = circular_left_shift(r_side, round_shifts[i])
        round_key = transpose(shifted_left + shifted_right, PC2)
        round_keys_arr.append(round_key)
        l_side = shifted_left
        r_side = shifted_right
    return round_keys_arr


'''F-function'''


# Split 48 bits into 6 bits each
def split_in_6bits(bits48):
    list_of_6bits = textwrap.wrap(bits48, 6)
    return list_of_6bits


# Returns first and last bit from a binary string
def get_first_and_last_bit(bits6):
    twobits = bits6[0] + bits6[-1]
    return twobits


# Returns a binary string except first and last bit
def get_middle_four_bit(bits6):
    fourbits = bits6[1:5]
    return fourbits


# Convert binary to decimal
def binary_to_decimal(binarybits):
    decimal = int(binarybits, 2)
    return decimal


# Convert decimal to binary
def decimal_to_binary(decimal):
    binary4bits = bin(decimal)[2:].zfill(4)
    return binary4bits


# Search for number in S-box
def sbox_lookup(sboxcount, first_last, middle4):
    d_first_last = binary_to_decimal(first_last)
    d_middle = binary_to_decimal(middle4)

    sbox_value = SBOX[sboxcount][d_first_last][d_middle]
    return decimal_to_binary(sbox_value)


# This is main function to perform function F
def functionF(pre32bits, key48bits):
    result = ""
    expanded_left_half = transpose(pre32bits, EXPANSION_TABLE)
    xor_value = XOR_to_use.getval(split(expanded_left_half), split(key48bits))
    bits6list = split_in_6bits(xor_value)
    for sboxcount, bits6 in enumerate(bits6list):
        first_last = get_first_and_last_bit(bits6)
        middle4 = get_middle_four_bit(bits6)
        sboxval = sbox_lookup(sboxcount, first_last, middle4)
        result += sboxval
    final32bits = transpose(result, PERMUTATION_TABLE)
    return final32bits


# encryption|decryption
# Split word by chars
def split(word):
    return [int(char) for char in word]


# Function for encryption|decryption
# action - E: encrypt, D: decrypt
def decrypt_encrypt(action, text, keys):
    text = text_by_blocks(text)
    transp_block_of_msg = []
    for b in text:
        transp_block_of_msg.append(transpose(b, INITIAL_PERMUTATION_TABLE))
    cipher = ''
    for p_plaintext in transp_block_of_msg:
        L, R = split_msg_in_half(p_plaintext)
        for round in range(NUM_OF_ROUNDS):
            if action == 'E':
                newR = XOR_to_use.getval(split(L), split(functionF(R, keys[round])))
            elif action == 'D':
                newR = XOR_to_use.getval(split(L), split(functionF(R, keys[15 - round])))
            newL = R
            R = newR
            L = newL
        cipher += transpose(R + L, INVERSE_PERMUTATION_TABLE)
    return cipher


# Permutation matrices
INITIAL_PERMUTATION_TABLE = ['58 ', '50 ', '42 ', '34 ', '26 ', '18 ', '10 ', '2',
                             '60 ', '52 ', '44 ', '36 ', '28 ', '20 ', '12 ', '4',
                             '62 ', '54 ', '46 ', '38 ', '30 ', '22 ', '14 ', '6',
                             '64 ', '56 ', '48 ', '40 ', '32 ', '24 ', '16 ', '8',
                             '57 ', '49 ', '41 ', '33 ', '25 ', '17 ', '9 ', '1',
                             '59 ', '51 ', '43 ', '35 ', '27 ', '19 ', '11 ', '3',
                             '61 ', '53 ', '45 ', '37 ', '29 ', '21 ', '13 ', '5',
                             '63 ', '55 ', '47 ', '39 ', '31 ', '23 ', '15 ', '7']

# Permutation matrices for the key
PC1 = [57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4]

PC2 = [14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10, 23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2, 41,
       52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32]

round_shifts = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

EXPANSION_TABLE = [32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17, 16,
                   17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1]

SBOX = [
    # Box-1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # Box-2

    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],

    # Box-3

    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]

    ],

    # Box-4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],

    # Box-5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # Box-6

    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]

    ],
    # Box-7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # Box-8

    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]

]
# P permutation
PERMUTATION_TABLE = [16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
                     2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25]

INVERSE_PERMUTATION_TABLE = ['40 ', '8 ', '48 ', '16 ', '56 ', '24 ', '64 ', '32',
                             '39 ', '7 ', '47 ', '15 ', '55 ', '23 ', '63 ', '31',
                             '38 ', '6 ', '46 ', '14 ', '54 ', '22 ', '62 ', '30',
                             '37 ', '5 ', '45 ', '13 ', '53 ', '21 ', '61 ', '29',
                             '36 ', '4 ', '44 ', '12 ', '52 ', '20 ', '60 ', '28',
                             '35 ', '3 ', '43 ', '11 ', '51 ', '19 ', '59 ', '27',
                             '34 ', '2 ', '42 ', '10 ', '50 ', '18 ', '58 ', '26',
                             '33 ', '1 ', '41 ', '9 ', '49 ', '17 ', '57 ', '25']

# main
opened_text = input('Enter the plain text: ')
print("\nPlain text text is:", opened_text)

start = time.time()

padded_msg = pkcs7(opened_text)
print('Plain text after PKCS#7:', padded_msg)

padded_bin = to_bin_str(padded_msg)


key = random_string()
print("Key is: ", key)

bin_key_64 = to_bin_str(key)
key_56bits = transpose(bin_key_64, PC1)
left_side, right_side = split_key_in_half(key_56bits)
roundkeys = round_keys(key_56bits)


print('\nEncryption: ')
encrypt = decrypt_encrypt('E', padded_bin, roundkeys)
print('Encrypted message is: ', bit_array_to_string(encrypt))

print('\nDecryption: ')
decrypt = decrypt_encrypt('D', encrypt, roundkeys)
print('Plain text is: ', unpad(bit_array_to_string(decrypt)))

XOR_to_use.ses_close()

end = time.time()
print('\nEncryption time is: ', end-start)
exit(0)
