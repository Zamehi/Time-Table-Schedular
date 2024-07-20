# Time-Table-Schedular
Applying Genetic Algorithm to apply various constraints constructing a time table schedular.

AI PROJECT 2024 – i210780
1.	Reading Excel Sheets:
a.	The code reads two sheets from an Excel file using Pandas' read_excel function.
b.	For each sheet, it skips the first row (which appears to contain headers) and then fills any NaN (Not a Number) values with an empty string.
c.	The fillna("") method is used to fill NaN values with an empty string.
d.	The first row is then used to set the column names.
2.	Extracting Data:
a.	After reading each sheet, specific columns are extracted from the DataFrame using indexing and stored as NumPy arrays.
b.	These columns include information about courses, such as course numbers, section numbers, instructor numbers, status, and strength.
c.	The .to_numpy() method is used to convert the DataFrame to a NumPy array.
d.	The extracted columns are concatenated together using np.concatenate() to combine the data from both sheets into a single array named listTeachers.
3.	Understanding the Data:
a.	The data extracted from the Excel sheets likely represents information about courses, sections, instructors, and their respective attributes.
b.	Each row in the listTeachers array likely corresponds to a specific course or section, with columns representing different attributes such as course number, section number, instructor number, status, and strength.
4.	Checking Data Type:
a.	Finally, the code prints the type of listTeachers using the type() function.
5.	Converting to Binary:
a.	The arr_to_bin function converts the listTeachers array to binary representation using NumPy's np.binary_repr function. This likely converts numeric values in the array to their binary representation.
6.	Definition:
a.	timeSlots: List of time slots representing different periods throughout the day.
b.	roomNumbers: List of room numbers or identifiers.
c.	roomSeats: List of corresponding seat capacities for each room.
d.	roomSeatsAvailability: Array indicating the availability of seats in each room. It's initially set based on seat capacities.
e.	roomCount: Total number of rooms.
f.	row: Number of rows in the timetable, likely representing the number of rooms.
g.	slots: Number of time slots per day.
h.	numDays: Number of days in the timetable.
i.	col: Total number of columns in the timetable, calculated based on the number of days and time slots.
7.	Generating Timetable:
a.	The generate_random_timetable function generates a random timetable based on the given constraints.
b.	It iterates over each room and time slot combination and randomly assigns courses or lectures to each slot while adhering to specific rules. For example:
i.	It ensures that lab sessions are not scheduled on consecutive time slots.
ii.	It randomly selects days for each course, ensuring that no course is scheduled on the same or adjacent days.
iii.	It prints out the scheduled entries for debugging purposes.
8.	Generating Population:
a.	The generate_population function creates a population of timetables. It iterates a given number of times (size) and generates a random timetable for each iteration using the generate_random_timetable function. The resulting timetables are stored in a list called population.
9.	Fitness Calculation:
a.	The calculate_fitness function assesses the fitness of a given timetable. It penalizes the timetable based on certain criteria:
i.	Penalizes if the same course is scheduled on consecutive or adjacent time slots.
ii.	Penalizes if a room is assigned for two different sections at the same time.
iii.	The fitness value is adjusted based on the penalties.
b.	The fitness function calculates the fitness for each timetable in the population by calling the calculate_fitness function iteratively. It then normalizes the fitness values by dividing each fitness value by the total fitness of all timetables in the population.
10.	Population Fitness:
a.	The fitness values for each timetable in the population are calculated using the fitness function.
b.	It prints out the normalized fitness values for each timetable in the population.
11.	Converting to NumPy Array:
a.	Before calculating fitness, the code checks if any timetable in the population is a Pandas DataFrame. If so, it converts it to a NumPy array. This step is essential for uniform handling of data during fitness calculation.

1.	Finding Maximum Fitness:
•	It iterates through the selected individuals (assuming selected is a list containing fitness values of individuals). For each selected individual, it finds the maximum fitness value (max_fitness) and its index (max_fitness_indices) within that individual.
2.	Selecting Individuals:
•	It selects a certain number of individuals (in this case, 2) based on their maximum fitness indices. For each selected individual, it retrieves the corresponding timetable from the population and adds it to the chosen_population list.
3.	Conversion to NumPy Array:
•	Finally, it converts the chosen_population list into a NumPy array.
4.	Output:
•	It prints a message indicating that the selection process is done and also prints the shape of the first timetable in the chosen population.
1.	Crossover Function (crossover):
•	This function takes the chosen population (parents), a list of teachers, and a mutation rate as input.
•	It performs crossover between pairs of parents to generate offspring.
•	The crossover point is randomly selected, and child timetables are created by combining parts of the parents' timetables.
•	After crossover, mutation is applied to the offspring timetables.
•	The function returns the offspring population.
2.	Mutation Function (mutate_timetable):
•	This function takes a timetable and a mutation rate as input.
•	It creates a copy of the timetable and applies mutation to it with the given probability.
•	In this case, mutation involves randomly changing the course assigned to certain cells in the timetable.
•	The function returns the mutated timetable.
3.	Next Generation Selection (select_next_generation):
•	This function selects individuals (timetables) for the next generation based on their fitness values.
•	It sorts individuals based on fitness in descending order and selects the top individuals.
•	The number of individuals to select for the next generation is determined by the num_selected parameter.
•	The function returns the selected individuals for the next generation.
4.	Back to Original Timetable Representation:
•	After selecting the next generation, the code converts the binary representation of timetables back to their original string representation.
•	It iterates over each cell in the timetables, converts binary strings to their corresponding course information using predefined dictionaries, and stores the converted data in a list.
•	The converted data is then used to create a DataFrame, where the slots list is added as the first row and room numbers are added as the first column.
•	Finally, the DataFrame is saved to an Excel file.


  
