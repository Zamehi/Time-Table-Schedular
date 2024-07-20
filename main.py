from PIL._imaging import display
from numpy.random import randint
import object
#import function

import numpy as np
import pandas as pd


# ----------------------- FUNCTION -------------------------



numberOfClasses = 0
numberOfSections = 30
numberOfRooms = 38
numberOfLabs = 16
rows = 54 # total number of classes
cols = 6 # time slots


# **************************** reading 1 *****************************

# Creating a 2D list of Cell objects with dimensions 3x3 using nested list comprehension
timetable_array = [[object.timetable(5) for col in range(16)] for row in range(38)]

# Read course teachers into a dictionary
teacher = pd.read_excel('C:\\Users\\Zunaira\\Downloads\\Courses Allocation, FSC, FAST-NUCES, Islamabad, Spring-2024.xlsx', sheet_name='Computing-Theory', skiprows=1)
teachers = teacher.fillna("")
teacher.columns = teacher.iloc[0]
teachers.head()

listTeachers = teacher[["Course Number", "Section Number", "Course Instructor Number", "Status", "Strength"]].to_numpy()
listTeachers = listTeachers[2:]
#print (listTeachers)




# **************************** reading 2 *****************************

# Read Lab teachers into a dictionary
labTeacher = pd.read_excel('C:\\Users\\Zunaira\\Downloads\\Courses Allocation, FSC, FAST-NUCES, Islamabad, Spring-2024.xlsx', sheet_name='Computing-Labs', skiprows=1)
labTeacher= labTeacher.fillna("")
labTeacher.columns = labTeacher.iloc[0]
labTeacher.head()

listTeachers2 = labTeacher[["Course Number", "Section Number", "Lab Course Instructor Number", "Status", "Strength"]].to_numpy()
listTeachers2 = listTeachers2[2:]
#print (listTeachers)

listTeachers = np.concatenate((listTeachers, listTeachers2))
print (type(listTeachers))# Function to convert each number in the array to binary

# ----------------------------------------------------------
# -------------------- Convert to BINARY -------------------
# ----------------------------------------------------------
def arr_to_bin(arr):
    return np.array([[np.binary_repr(num) for num in row] for row in arr])
listTeachers = arr_to_bin(listTeachers)


# *********************************************************
# *********************** DEFINITION **********************
# *********************************************************


timeSlots = ['08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15']

roomNumbers = ['C301', 'C302', 'C303', 'C304', 'lab1', 'lab2']
roomSeats = [ 60, 60, 60,60 , 120, 120]
roomSeatsAvailability  = np.where(roomSeats == 60, "free", roomSeats)
roomSeatsAvailability  = np.where(roomSeatsAvailability == 120, "free", roomSeatsAvailability)
roomCount = int(len(roomNumbers))
row = roomCount
slots = 6
numDays = 5
col = numDays * slots
print( (listTeachers))
print (row)


# *********************************************************
# ****************** GENERATE TIMETABLE *******************
# *********************************************************
import random
import numpy as np


def generate_random_timetable(listTeachers, row, col):
    timetable = []
    room_assignments = {}
    prev_entry = None  # Variable to store the previous entry
    course_days = {}

    # Generate a random timetable
    for i in range(int(row)):
        room = roomNumbers[i % len(roomNumbers)]  # Use room symbol as key, cycling through roomNumbers
        lab_count = 0  # Counter to track consecutive lab slots
        # for specifice room numbers
        for j in range(int(col)):
            if (i == 4 or i == 5):  # for lab rooms
                course_info = random.choice(listTeachers[10:15])
                while course_info[0] == '' or course_info[1] == '' or course_info[2] == '':
                    course_info = random.choice(listTeachers[10:15])
                course_code = str(course_info[0])
                section = str(course_info[1])
                instructor = str(course_info[2])
                status = str(course_info[3])
                strength = str(course_info[4])
                # 2 consecutive lab slots
                if lab_count == 1 and prev_entry:
                    # Append the previous entry
                    timetable.append(prev_entry)
                    lab_count = 0  # Reset lab_coun
                    continue
                if j%2 == 0:
                    lab_count+=1

            else :
                course_info = random.choice(listTeachers[0:10])
                while course_info[0] == '' or course_info[1] == '' or course_info[2] == '':
                    course_info = random.choice(listTeachers[0:10])
                course_code = str(course_info[0])
                section = str(course_info[1])
                instructor = str(course_info[2])
                status = str(course_info[3])
                strength = str(course_info[4])


            if (course_code, section) in course_days:
                # Get the days on which the course has been scheduled
                scheduled_days = course_days[(course_code, section)]
                # Randomly select a day that is not same or adjacent to any of the scheduled days
                available_days = [d for d in range(int(col)) if
                                  d not in scheduled_days and all(abs(d - sd) > 1 for sd in scheduled_days)]
                if available_days:
                    day = random.choice(available_days)
                    scheduled_days.append(day)
                else:
                    # If no suitable day found, skip scheduling this lecture
                    entry = '0' + ' -- ' +'0' + ' -- ' + '0' + ' -- ' + '0' + ' -- ' + '0'
                    print(i, j, entry)
                    timetable.append(entry)
                    prev_entry = entry  # Update prev_entry
                    room_assignments[(j, section)] = course_code
                    continue
            else:
                # If the course is not yet scheduled, select a random day
                day = random.randint(0, int(col) - 1)
                course_days[(course_code, section)] = [day]


            entry = course_code + ' -- ' + section + ' -- ' + instructor + ' -- ' + status + ' -- ' + strength
            print (i, j ,entry)
            timetable.append(entry)
            prev_entry = entry  # Update prev_entry
            room_assignments[(j, section)] = course_code

    # Reshape timetable into 2D array
    timetable = np.array(timetable).reshape((int(row), int(col)))

    return timetable


# *********************************************************
# ********************** POPULATION ***********************
# *********************************************************
def generate_population(size, listTeachers, row, col):
    # size = 20
    population = []

    for i in range(size):
      df = pd.DataFrame(generate_random_timetable(listTeachers, row, col))
      df = pd.DataFrame(df)
      df.columns = ['08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15']
      population.append(df)

    return population


population = generate_population(100, listTeachers, row, col)
print (population[1])


# *********************************************************
# ****************** FITNESS FUNCTION   *******************
# *********************************************************



def calculate_fitness(timetable):
    fitness = 0
    # Convert Pandas DataFrame to NumPy array if needed
    if isinstance(timetable, pd.DataFrame):
        timetable = timetable.values

    # Penalize if the same course is scheduled on consecutive or adjacent slots
    for i in range(timetable.shape[0]):
        for j in range(timetable.shape[1] - 1):
            if isinstance(timetable[i, j], str) and isinstance(timetable[i, j + 1], str):
                # Split the string and check for consecutive courses
                if timetable[i, j].split(' -- ')[0] == timetable[i, j + 1].split(' -- ')[0]:
                    fitness -= 1

    # Penalize if a room is assigned for two different sections at the same time
    room_assignments = {}  # Dictionary to store room assignments
    for i in range(timetable.shape[0]):
        for j in range(timetable.shape[1]):
            if isinstance(timetable[i, j], str):
                course_info = timetable[i, j].split(' -- ')
                section = course_info[1]
                room = course_info[-2]  # Assuming the second last item in the course_info represents the room
                seats_needed = int(course_info[-1])  # Assuming the last item in the course_info represents the section size

                # Check if the room is free at this time slot
                if (j, room) in room_assignments:
                    fitness -= 1  # Penalize if the room is already assigned to a different section at the same time
                else:
                    room_assignments[(j, room)] = section

                # Check if the room has enough seats for the section
                if room in roomSeats:
                    if roomSeats[room] < seats_needed:
                        fitness -= 1  # Penalize if the room doesn't have enough seats for the section
                else:
                    fitness -= 1  # Penalize if the room is not in the roomSizes dictionary

    return fitness

def fitness(population):
    timetable_fitness = []
    for i in range(len(population)):
        timetable_fitness.append(calculate_fitness(population[i]))
    # Calculate the sum of all elements in the array
    total_fitness = np.sum(timetable_fitness)
    for i in range(len(timetable_fitness)):
        timetable_fitness[i] = round(timetable_fitness[i] / total_fitness, 5)
    print(timetable_fitness)
    return timetable_fitness



# Converting to numpy array
for i in range(100):
  if isinstance(population[i], pd.DataFrame):
    population[i] = population[i].values
#print ("Population is ",population)

timetable_fitness = fitness(population)
print ("Fitness Calculated...")




# *********************************************************
# ****************** SELECTION FUNCTION *******************
# *********************************************************
# Assuming population is a list of NumPy arrays
selected = []
for i in range (4):
  # Select 8 individuals randomly
  selected_individuals = random.sample(timetable_fitness, 8)
  #print(timetable_fitness)

  # Get the indices of selected individuals
  selected_indices = [timetable_fitness.index(individual) for individual in selected_individuals]
  #print(selected_indices)

  # Print the selected individuals
  for i, individual in enumerate(selected_individuals, 1):
    #print("Individual {}: {}".format(i, individual))
    selected.append(selected_individuals)

  #print(selected)

# ---------- find max from each selected tuples --------------
# --------------------- TOURNAMENT ---------------------------
max_fitness = []
max_fitness_indices =[]
for i in range(4):
  max_fitness.append(max(selected[i]))
  # Get the index of the maximum value
  max_fitness_indices.append(np.argmax(selected[i]))

# selecred 4 individuals
#print (type(max_fitness))
#print (max_fitness_indices)
max_fitness_indices = np.array(max_fitness_indices)
max_fitness = np.array(max_fitness)
population = np.array(population)

chosen_population = []
for i in range (2):
  chosen_population.append(population[max_fitness_indices[i]])
# Converting chosen population from list to numpy array
chosen_population= np.array(chosen_population)

#print (chosen_population)
#print (type(chosen_population))
print ("Selection Done...")
print ((chosen_population[0].shape))
# selected = population[max_indices[i]]


# *********************************************************
# ****************** CROSSOVER FUNCTION *******************
# *********************************************************

# chosen population[] has four timetables

'''
# Function to perform crossover
def crossover(time_table, mutation_rate):
    num_tables = len(chosen_population)
    offspring = []

    # Perform crossover for each pair of tables
    for i in range(num_tables // 2):
        # Randomly select a crossover point
        crossover_point = 4

        # Create offspring by exchanging rows at the crossover point
        child1 = np.vstack((time_table[2 * i][:crossover_point], time_table[2 * i + 1][crossover_point:]))
        child2 = np.vstack((time_table[2 * i + 1][:crossover_point], time_table[2 * i][crossover_point:]))

        # Apply mutation to offspring
        child1_mutated = mutate(child1, mutation_rate)
        child2_mutated = mutate(child2, mutation_rate)

        offspring.extend([child1_mutated, child2_mutated])

    return np.array(offspring)

def mutate(individual, mutation_rate):
    mutated_individual = np.copy(individual)

    # Iterate through each gene and apply mutation based on the mutation rate
    for row in range(mutated_individual.shape[0]):
        for col in range(mutated_individual.shape[1]):
            if np.random.rand() < mutation_rate:
                # Apply mutation (e.g., randomly change the value of the gene)
                mutated_individual[row, col] = np.random.randint(0, 10)  # Example mutation

    return mutated_individual

'''


print (chosen_population.shape)

import numpy as np
import random


def crossover(chosen_population, listTeachers, mutation_rate):
    population_size = chosen_population.shape[0]
    num_days = chosen_population.shape[1]
    num_rooms = chosen_population.shape[2]
    offspring = []

    while len(offspring) < population_size:
        # Randomly select two parents
        parent_indices = random.sample(range(population_size), 2)
        parent1, parent2 = chosen_population[parent_indices[0]], chosen_population[parent_indices[1]]

        # Perform crossover
        crossover_point = random.randint(1, num_days - 2)
        child = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)

        # Apply mutation
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                if random.random() < mutation_rate:
                    course_info = random.choice(listTeachers)
                    while course_info[0] == '' or course_info[1] == '' or course_info[2] == '':
                        course_info = random.choice(listTeachers)
                    child[i, j] = f"{course_info[0]} -- {course_info[1]} -- {course_info[2]}"

        offspring.append(child)

    # Combine parents and offspring to form the new population
    population = np.array(offspring)

    return population


# Perform crossover on the given time table
offspring = crossover(chosen_population, listTeachers,0.1)
print ("Crossover Done...")
print ((offspring))

# Again calculate fitness
timetable_fitness2 = fitness(offspring)
# finding maximum
max_timetable_fitness2 = np.max(timetable_fitness2)
max_timetable_fitness2_index = np.argmax(timetable_fitness2)
print (max_timetable_fitness2_index)

selected_offspring = offspring[max_timetable_fitness2_index]



# *********************************************************
# ****************** MUTATION  FUNCTION *******************
# *********************************************************


def generate_random_course():
    # Randomly select a course from the list
    return random.choice(listTeachers)

def mutate_timetable(timetable, mutation_rate):
    mutated_timetable = np.copy(timetable)

    # Iterate through each cell in the timetable
    for i in range(mutated_timetable.shape[0]):
        for j in range(mutated_timetable.shape[1]):
            # Apply mutation with the given probability
            if np.random.rand() < mutation_rate:
                # Perform mutation operation here (e.g., randomly change the course)
                # Example: randomly select a new course from a predefined list
                mutated_timetable[i, j] = generate_random_course()

    return mutated_timetable



# Perform mutation on offspring
mutated_offspring = []
for timetable in offspring:
        mutated_timetable = mutate_timetable(timetable, 0.2)
        mutated_offspring.append(mutated_timetable)

mutated_offspring= np.array(mutated_offspring)

print ("Mutation Done...")
print ((mutated_offspring[1].shape))

# *********************************************************
# ****************** NEXT GENERATION   ********************
# *********************************************************

def select_next_generation(mutated_population, fitness_values, num_selected):
    # Sort individuals based on fitness (descending order)
    sorted_indices = np.argsort(fitness_values)[::-1]
    # Select the first individual (highest fitness) for the next generation
    next_generation = [mutated_population[sorted_indices[0]]]
    return next_generation

# Evaluate FITNESS of offspring
offspring_fitness = []
for timetable in mutated_offspring:
    fitness_value = fitness(mutated_offspring)
    offspring_fitness.append(fitness_value)

offspring_fitness = np.array(offspring_fitness)
print(offspring_fitness.shape)
num_selected = 1  # Number of individuals to select for the next generation
next_generation = select_next_generation(mutated_offspring, offspring_fitness, num_selected)
next_generation= np.array(next_generation[0][0])
print ("Next Generation...")

# Define dictionaries to map binary representations to their respective strings
course_map = {'1': 'Object Oriented Programming (CS)',
              '10': 'Object Oriented Programming (AI)',
              '11': 'Object Oriented Programming (DS)',
              '100': 'Database Systems (CS)',
              '101': 'Digital Logic Design (CS)',
              '111': 'Database Systems Lab (DS)',
              '110': 'Object Oriented Programming Lab (CS)',#6
              '1001': 'Programming Fundamentals (CS)',#9
              '1000': 'Artificial Intelligence Lab (AI)',
              '0':'Null'}

section_map = {'1': 'BCS-2A',
               '10': 'BCS-2B',
               '11': 'BCS-2C',
               '100': 'BCS-2D',
               '101': 'BCS-2E',
               '110': 'BCS-2F',
               '111': 'BCS-2G',
               '0':'Null'}

instructor_map = {'1': 'Dr. Ali Zeeshan Ijaz',
                  '10': 'Mr. Shehreyar Rashid',
                  '11': 'Ms. Marium Hida',
                  '101': 'Mr. Amir Gulzar',
                  '110': 'Ms. Bushra Fatima Tariq',
                  '111': 'Mr. Usama Bin Imran',
                  '0':'Null'}

status_map = {'0': 'Course', '1': 'Lab'}
strength_map = {'110010':'50', '0':'Null'}
converted_data_list = []

''''
for i in range(next_generation.shape[0]):
    for j in range(next_generation.shape[1]):
        next_generation[i][j] = np.array(next_generation[i][j])
'''

def convert_to_string(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.astype(str)
    elif isinstance(arr, int):
        arr = '0 -- 0 -- 0 -- 0 -- 0'
    arr_str = ' -- '.join(arr)
    return arr_str
for i in range(next_generation.shape[0]):
    for j in range(next_generation.shape[1]):
        print(i, " ", j, "->", (type(next_generation[i][j])))

for i in range(next_generation.shape[0]):
    for j in range(next_generation.shape[1]):
        if not isinstance(next_generation[i][j], str):
            next_generation[i][j] =convert_to_string(next_generation[i][j])

print (next_generation.shape)
# Convert all elements to strings
#next_generation = np.array([[str(item) for item in row] for row in next_generation])
#for i in range(next_generation.shape[0]):
#    for j in range(next_generation.shape[1]):
#        print(i, " ", j, "->", ((next_generation[i][j])))

print (next_generation.shape)

import numpy as np

# Initialize the list to store converted data
converted_data_list = []

# Iterate over each row of next_generation
for row in next_generation:
    if isinstance(row, np.ndarray):
        # Process each element of the NumPy array
        for item in row:
            # Convert the item to string and split into components
            components = item.split(" -- ")

            # If there are exactly 5 components
            if len(components) == 5:
                # Convert binary components to strings using the dictionaries
                course_number = course_map.get(components[0], 'Unknown')
                section_number = section_map.get(components[1], 'Unknown')
                instructor_number = instructor_map.get(components[2], 'Unknown')
                status = status_map.get(components[3], 'Unknown')
                strength = strength_map.get(components[4], 'Unknown')

                # Combine the components into an array
                converted_data = [course_number, section_number, instructor_number, status, strength]

                # Append converted_data to the list
                converted_data_list.append(converted_data)
    else:
        # Convert the row (which is a string) into components
        components = row.split(" -- ")

        # If there are exactly 5 components
        if len(components) == 5:
            # Convert binary components to strings using the dictionaries
            course_number = course_map.get(components[0], 'Unknown')
            section_number = section_map.get(components[1], 'Unknown')
            instructor_number = instructor_map.get(components[2], 'Unknown')
            status = status_map.get(components[3], 'Unknown')
            strength = strength_map.get(components[4], 'Unknown')

            # Combine the components into an array
            converted_data = [course_number, section_number, instructor_number, status, strength]

            # Append converted_data to the list
            converted_data_list.append(converted_data)
converted_data_list = np.array(converted_data_list)

for i in range(converted_data_list.shape[0]):
        print(i, " ", "->", ((converted_data_list[i])))
print ((converted_data_list.shape))


# Function to convert an individual item to its string representation
def convert_item_to_string(item):
    if isinstance(item, str):
        components = item.split(" -- ")
        if len(components) == 5:
            course_number = course_map.get(components[0], 'Unknown')
            section_number = section_map.get(components[1], 'Unknown')
            instructor_number = instructor_map.get(components[2], 'Unknown')
            status = status_map.get(components[3], 'Unknown')
            strength = strength_map.get(components[4], 'Unknown')
            return f"{course_number} -- {section_number} -- {instructor_number} -- {status} -- {strength}"
    return item
# Apply conversion to each element of the next_generation array
converted_next_generation = np.vectorize(convert_item_to_string)(next_generation)
print ("Converted again ",converted_next_generation.shape)
# *********************************************************
# ****************** BACK TO ORIGINAL  ********************
# *********************************************************

import pandas as pd

# Create DataFrame from converted_data_list
df = pd.DataFrame(converted_next_generation)

# Add slots list as the first row

df.insert(0, 'Slots', slots)

# Add room numbers as the first column

#timeSlots = ['08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15','08:30-09:50', '10:00-11:20', '11:30-12:50', '01:00-02:20', '02:30-03:50', '03:55-05:15']

#roomNumbers = ['C301', 'C302', 'C303', 'C304', 'lab1', 'lab2']
roomNumbers = ['C301', 'C302', 'C303', 'C304', 'lab1', 'lab2'] * (len(df) // len(roomNumbers)) + roomNumbers[:len(df) % len(roomNumbers)]
df.insert(0, 'RoomNo', roomNumbers)

# Save DataFrame to Excel
#"i210269_FaizanKaramat_ass02&03.ipynb"
df.to_excel('C:\\Users\\Zunaira\\Downloads\\timetable.xlsx', index=False)


from tabulate import tabulate

# Function to display a numpy array as a table
def display_table(array, day):
    headers = [f"Column {i+1}" for i in range(array.shape[1])]
    print(f"Timetable for {day}:")
    print(tabulate(array, headers=headers, tablefmt="grid"))

# Assuming 'population' is a numpy array of shape (6, 30)
population = np.random.randint(0, 10, (6, 30))  # Example population array

# Divide the population array into five separate arrays, each representing a day
monday = converted_next_generation[:, :6]
tuesday = converted_next_generation[:, 6:12]
wednesday = converted_next_generation[:, 12:18]
thursday = converted_next_generation[:, 18:24]
friday = converted_next_generation[:, 24:]


# Display each day's timetable
display_table(monday, "Monday")
display_table(tuesday, "Tuesday")
display_table(wednesday, "Wednesday")
display_table(thursday, "Thursday")
display_table(friday, "Friday")