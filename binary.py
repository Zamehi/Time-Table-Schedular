# --------------------- REVERSE BINARY ----------------------
reverse_binary_lab_codes = {binary_code: lab_number for lab_number, binary_code in binary_lab_codes.items()}
# Example: Convert binary code back to lab number
binary_code = '01111'  # Example binary code
lab_number = reverse_binary_lab_codes.get(binary_code, 'Unknown')
print(f'Lab number corresponding to binary code {binary_code}: {lab_number}')


# Convert array to binary code using numpy.packbits()
binary_code = np.packbits(listTeachers.astype(int))

print("Binary code:", binary_code)